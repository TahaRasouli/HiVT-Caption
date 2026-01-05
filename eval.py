import os
import torch
import numpy as np
from argparse import ArgumentParser
import pytorch_lightning as pl
from tqdm import tqdm

from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.hivt import HiVT
from nuscenes.map_expansion.map_api import NuScenesMap

# Speed boost for A6000
torch.set_float32_matmul_precision('high')

class AdvancedEvaluator:
    def __init__(self, nusc_root: str):
        # Maps are located in the 'maps' folder under the v1.0-trainval parent directory
        # The path provided: /mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta/v1.0-trainval/
        self.map_root = os.path.dirname(nusc_root.rstrip('/'))
        self.map_names = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
        self.maps = {
            loc: NuScenesMap(dataroot=self.map_root, map_name=loc) 
            for loc in self.map_names
        }

    def is_off_road(self, pt, city):
        """Checks if a global (x, y) point is outside drivable areas."""
        if city not in self.maps: 
            return False
        
        # CORRECTED METHOD NAME
        layers = self.maps[city].layers_on_point(pt[0], pt[1])
        
        # In NuScenes, layers_on_point returns a dict where keys are layer names
        # and values are the tokens of the objects at that point.
        drivable = layers.get('drivable_area', '')
        lane = layers.get('lane', '')
        lane_conn = layers.get('lane_connector', '')
        
        return not (drivable or lane or lane_conn)

    def compute_scene_metrics(self, batch, y_hat):
        """Calculates advanced realism metrics."""
        # Selection logic remains the same
        reg_mask = ~batch['padding_mask'][:, 20:]
        l2_norm = (torch.norm(y_hat[:, :, :, :2] - batch.y, p=2, dim=-1) * reg_mask).sum(dim=-1)
        best_mode_idx = l2_norm.argmin(dim=0)
        best_trajs = y_hat[best_mode_idx, torch.arange(batch.num_nodes), :, :2]

        # 1. Heading Error (Rad)
        # Using .detach() to ensure no grad tracking during eval
        v_pred = (best_trajs[:, -1] - best_trajs[:, -2]).detach()
        v_gt = (batch.y[:, -1] - batch.y[:, -2]).detach()
        
        # Compute angles
        pred_theta = torch.atan2(v_pred[:, 1], v_pred[:, 0])
        gt_theta = torch.atan2(v_gt[:, 1], v_gt[:, 0])
        
        heading_err = torch.abs(pred_theta - gt_theta)
        # Wrap to [0, pi]
        heading_err = torch.where(heading_err > np.pi, 2*np.pi - heading_err, heading_err)

        # 2. Off-Road Rate
        off_road_count = 0
        total_agents = best_trajs.size(0)
        
        # Map city name index to actual string if necessary
        # NuScenes usually stores location in the batch.
        # If batch.city is a list of strings:
        city = batch.city[0] if hasattr(batch, 'city') else 'singapore-onenorth'
        
        origin = batch.origin[0].cpu().numpy()
        theta = batch.theta[0].cpu().numpy()
        
        cos, sin = np.cos(-theta), np.sin(-theta)
        rot_mat = np.array([[cos, -sin], [sin, cos]])

        for i in range(total_agents):
            local_dest = best_trajs[i, -1].cpu().numpy()
            # Transform local destination back to global
            global_dest = (local_dest @ rot_mat.T) + origin
            
            if self.is_off_road(global_dest, city):
                off_road_count += 1

        # 3. Collision Rate
        collision_detected = 0
        # Check every 0.5s (every 5 steps at 10Hz) to save time
        for t in range(0, best_trajs.size(1), 5):
            dists = torch.cdist(best_trajs[:, t, :], best_trajs[:, t, :])
            dists += torch.eye(total_agents, device=best_trajs.device) * 10.0
            if (dists < 1.8).any(): # Using 1.8m as average car width threshold
                collision_detected = 1
                break

        return {
            "heading_err": heading_err.mean().item(),
            "off_road": off_road_count / total_agents if total_agents > 0 else 0,
            "collision": collision_detected
        }

def main():
    parser = ArgumentParser()
    parser.add_argument("--nusc_root", type=str, default="/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta/v1.0-trainval/")
    parser.add_argument("--processed_root", type=str, default="/mount/studenten/projects/rasoulta/dataset")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Load Model
    model = HiVT.load_from_checkpoint(args.ckpt_path, strict=False)
    model.eval()

    # Setup Evaluator
    evaluator = AdvancedEvaluator(args.nusc_root)

    # Setup Data
    datamodule = NuScenesHiVTDataModule(root=args.processed_root, val_batch_size=args.batch_size)
    datamodule.setup(stage="validate")
    val_loader = datamodule.val_dataloader()

    metrics = {"heading": [], "off_road": [], "collision": []}

    print("--- Starting Advanced Inference ---")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch = batch.to(model.device)
            y_hat, pi = model(batch)
            
            scene_res = evaluator.compute_scene_metrics(batch, y_hat)
            metrics["heading"].append(scene_res["heading_err"])
            metrics["off_road"].append(scene_res["off_road"])
            metrics["collision"].append(scene_res["collision"])

    print("\n" + "="*50)
    print("ADVANCED REALISM METRICS")
    print("-" * 50)
    print(f" • Heading Error (Rad): {np.mean(metrics['heading']):.4f}")
    print(f" • Off-Road Rate (%)  : {np.mean(metrics['off_road'])*100:.2f}%")
    print(f" • Collision Rate (%) : {np.mean(metrics['collision'])*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()