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
        if city not in self.maps: return False
        # get_layers_on_point returns a list of layer names at that coordinate
        layers = self.maps[city].get_layers_on_point(pt[0], pt[1])
        # A point is "on-road" if it's in a drivable_area or a lane
        return not ('drivable_area' in layers or 'lane' in layers)

    def compute_scene_metrics(self, batch, y_hat):
        """
        Calculates advanced realism metrics for a batch.
        y_hat: [modes, nodes, steps, 2]
        """
        # 1. Best Mode Selection (based on minFDE)
        reg_mask = ~batch['padding_mask'][:, 20:]
        l2_norm = (torch.norm(y_hat[:, :, :, :2] - batch.y, p=2, dim=-1) * reg_mask).sum(dim=-1)
        best_mode_idx = l2_norm.argmin(dim=0)
        best_trajs = y_hat[best_mode_idx, torch.arange(batch.num_nodes), :, :2] # [N, 30, 2]

        # 2. Final Heading Error (Rad)
        # Using the last two points to approximate the heading vector
        v_pred = best_trajs[:, -1] - best_trajs[:, -2]
        v_gt = batch.y[:, -1] - batch.y[:, -2]
        heading_err = torch.abs(torch.atan2(v_pred[:, 1], v_pred[:, 0]) - 
                                torch.atan2(v_gt[:, 1], v_gt[:, 0]))
        # Wrap angles to [0, pi]
        heading_err = torch.min(heading_err, 2*np.pi - heading_err)

        # 3. Off-Road Rate
        # We need to transform local back to global using batch.origin and batch.theta
        off_road_count = 0
        total_agents = best_trajs.size(0)
        
        # Note: In HiVT TemporalData, city is often stored as an attribute
        city = getattr(batch, 'city', ['singapore-onenorth'])[0]
        origin = batch.origin[0].cpu().numpy() # [1, 2]
        theta = batch.theta[0].cpu().numpy()
        
        # Inverse Rotation Matrix
        cos, sin = np.cos(-theta), np.sin(-theta)
        rot_mat = np.array([[cos, -sin], [sin, cos]])

        for i in range(total_agents):
            local_path = best_trajs[i].cpu().numpy()
            global_path = (local_path @ rot_mat.T) + origin
            
            # Check the final destination (t=3s) for off-road violation
            if self.is_off_road(global_path[-1], city):
                off_road_count += 1

        # 4. Collision Rate (Proximity check)
        # Check if any two agents' best-trajs are closer than 2.0m at any future step
        collision_detected = 0
        for t in range(best_trajs.size(1)):
            dists = torch.cdist(best_trajs[:, t, :], best_trajs[:, t, :])
            # Set diagonal to high value
            dists += torch.eye(total_agents, device=best_trajs.device) * 10.0
            if (dists < 2.0).any():
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
    parser.add_argument("--processed_root", type=str, default="/mount/arbeitsdaten/studenten4/rasoulta")
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