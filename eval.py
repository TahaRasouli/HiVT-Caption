import os
import torch
import numpy as np
from argparse import ArgumentParser
import pytorch_lightning as pl
from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.hivt import HiVT
from nuscenes.map_expansion.map_api import NuScenesMap

# Speed up A6000
torch.set_float32_matmul_precision('high')

class AdvancedEvaluator:
    def __init__(self, dataroot):
        self.dataroot = dataroot
        self.maps = {
            loc: NuScenesMap(dataroot=self.dataroot, map_name=loc) 
            for loc in ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
        }

    def compute_off_road_rate(self, trajs, city, origin, theta):
        """
        Checks if the best predicted trajectory leaves the drivable area.
        trajs: [N, T, 2] (Local coordinates)
        """
        n_usc_map = self.maps[city]
        # Transform back to global for map query
        cos, sin = np.cos(-theta), np.sin(-theta)
        rot_mat = np.array([[cos, -sin], [sin, cos]])
        
        off_road_count = 0
        total_agents = trajs.shape[0]
        
        for i in range(total_agents):
            # Convert local traj back to global
            global_traj = (trajs[i] @ rot_mat.T) + origin
            
            for pt in global_traj[::5]: # Sample every 0.5s for speed
                layers = n_usc_map.get_layers_on_point(pt[0], pt[1])
                if 'drivable_area' not in layers and 'lane' not in layers:
                    off_road_count += 1
                    break
        return off_road_count / total_agents if total_agents > 0 else 0

    def compute_collision_rate(self, all_trajs, radius=2.0):
        """
        Checks if predicted trajectories of different agents collide.
        all_trajs: [N, T, 2]
        """
        N, T, _ = all_trajs.shape
        if N < 2: return 0
        
        collisions = 0
        for t in range(T):
            dist_matrix = torch.cdist(all_trajs[:, t, :], all_trajs[:, t, :])
            # Mask self-distances
            dist_matrix += torch.eye(N, device=all_trajs.device) * 100
            if (dist_matrix < radius).any():
                collisions += 1
                break
        return collisions / 1.0 # Returns 1 if scene has a collision, 0 otherwise

def evaluate():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()

    model = HiVT.load_from_checkpoint(args.ckpt_path, strict=False)
    model.eval()
    
    # Initialize Map Evaluator
    adv_eval = AdvancedEvaluator(os.path.join(args.root, '..')) # Adjust based on your folder struct

    datamodule = NuScenesHiVTDataModule(root=args.root, val_batch_size=args.batch_size)
    datamodule.setup(stage="validate")
    val_loader = datamodule.val_dataloader()

    # Trackers
    all_metrics = {"off_road": [], "collision": [], "angle_error": []}

    print("--- Running Advanced Evaluation ---")
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(model.device)
            y_hat, pi = model(batch) # y_hat: [6, N, 30, 2]
            
            # 1. Best Mode Selection (per agent)
            # Find best mode based on minFDE for the metric calculation
            reg_mask = ~batch['padding_mask'][:, 20:]
            l2_norm = (torch.norm(y_hat[:, :, :, :2] - batch.y, p=2, dim=-1) * reg_mask).sum(dim=-1)
            best_mode_idx = l2_norm.argmin(dim=0)
            best_trajs = y_hat[best_mode_idx, torch.arange(batch.num_nodes), :, :2]

            # 2. Final Angle Error
            v_pred = best_trajs[:, -1] - best_trajs[:, -2]
            v_gt = batch.y[:, -1] - batch.y[:, -2]
            angle_error = torch.abs(torch.atan2(v_pred[:, 1], v_pred[:, 0]) - 
                                    torch.atan2(v_gt[:, 1], v_gt[:, 0]))
            all_metrics["angle_error"].append(angle_error.mean().item())

            # 3. Collision Rate (Self-collisions in scene)
            all_metrics["collision"].append(adv_eval.compute_collision_rate(best_trajs))

            # 4. Off-Road Rate (Requires map)
            # origin and theta are stored in batch
            for i in range(batch.num_graphs):
                # Note: This logic assumes batch contains city/origin info
                # You may need to slice the batch per scene here
                pass 

    print("\n" + "="*40)
    print(f"Angle Error (Rad): {np.mean(all_metrics['angle_error']):.4f}")
    print(f"Collision Rate: {np.mean(all_metrics['collision']):.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate()