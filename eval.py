import os
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.hivt import HiVT

# Performance optimization for A6000
torch.set_float32_matmul_precision('high')

class RealismEvaluator:
    @staticmethod
    def compute_metrics(batch, y_hat):
        """
        Calculates Heading Error and Collision Rate.
        y_hat: [modes, nodes, steps, 2]
        """
        # --- 1. Best Mode Selection (MinFDE) ---
        reg_mask = ~batch['padding_mask'][:, 20:]
        l2_norm = (torch.norm(y_hat[:, :, :, :2] - batch.y, p=2, dim=-1) * reg_mask).sum(dim=-1)
        best_mode_idx = l2_norm.argmin(dim=0)
        best_trajs = y_hat[best_mode_idx, torch.arange(batch.num_nodes), :, :2]

        # --- 2. Heading Error (Rad) ---
        # Focus on primary agents (agent_index)
        agent_idx = batch.agent_index
        v_pred = (best_trajs[agent_idx, -1] - best_trajs[agent_idx, -2])
        v_gt = (batch.y[agent_idx, -1] - batch.y[agent_idx, -2])
        
        # Calculate yaw angles
        pred_yaw = torch.atan2(v_pred[:, 1], v_pred[:, 0])
        gt_yaw = torch.atan2(v_gt[:, 1], v_gt[:, 0])
        
        heading_err = torch.abs(pred_yaw - gt_yaw)
        # Wrap angle difference to [0, pi]
        heading_err = torch.where(heading_err > np.pi, 2*np.pi - heading_err, heading_err)

        # --- 3. Collision Rate ---
        # Checks if any two agents in the scene are predicted to be closer than 1.8m
        collision_detected = 0
        # Check at 1.5s and 3.0s (midpoint and endpoint)
        for t in [14, 29]: 
            dists = torch.cdist(best_trajs[:, t, :], best_trajs[:, t, :])
            # Ignore self-distance
            dists += torch.eye(best_trajs.size(0), device=best_trajs.device) * 10.0
            if (dists < 1.8).any(): 
                collision_detected = 1
                break

        return heading_err.mean().item(), collision_detected

def main():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="/mount/arbeitsdaten/studenten4/rasoulta")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Load Model
    model = HiVT.load_from_checkpoint(args.ckpt_path, strict=False)
    model.eval()
    model.cuda()
    
    # Setup Data
    dm = NuScenesHiVTDataModule(root=args.root, val_batch_size=args.batch_size)
    dm.setup(stage="validate")
    
    # Trackers for Realism
    h_errs, colls = [], []
    # Trackers for Accuracy
    ades, fdes, mrs = [], [], []
    
    print(f"--- Starting Inference on {args.ckpt_path} ---")
    with torch.no_grad():
        for batch in tqdm(dm.val_dataloader(), desc="Evaluating"):
            batch = batch.to(model.device)
            y_hat, _ = model(batch)
            
            # Realism Metrics
            h, c = RealismEvaluator.compute_metrics(batch, y_hat)
            h_errs.append(h)
            colls.append(c)
            
            # Standard Accuracy Metrics (Primary Agent only)
            agent_idx = batch.agent_index
            y_agent = batch.y[agent_idx]
            y_hat_agent = y_hat[:, agent_idx, :, :2]
            
            # FDE per mode
            fde_all = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
            best_m = fde_all.argmin(dim=0)
            y_best = y_hat_agent[best_m, torch.arange(len(agent_idx))]
            
            # ADE/FDE/MR calculation
            ade = torch.norm(y_best - y_agent, p=2, dim=-1).mean()
            fde = torch.norm(y_best[:, -1] - y_agent[:, -1], p=2, dim=-1).mean()
            mr = (torch.norm(y_best[:, -1] - y_agent[:, -1], p=2, dim=-1) > 2.0).float().mean()
            
            ades.append(ade.item())
            fdes.append(fde.item())
            mrs.append(mr.item())

    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("-" * 50)
    print(f" • MinADE         : {np.mean(ades):.4f} m")
    print(f" • MinFDE         : {np.mean(fdes):.4f} m")
    print(f" • Miss Rate      : {np.mean(mrs)*100:.2f} %")
    print("-" * 50)
    print(f" • Heading Error  : {np.mean(h_errs):.4f} rad ({np.degrees(np.mean(h_errs)):.2f}°)")
    print(f" • Collision Rate : {np.mean(colls)*100:.2f} %")
    print("="*50)

if __name__ == "__main__":
    main()