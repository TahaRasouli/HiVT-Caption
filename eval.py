import os
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.hivt import HiVT

# Optimize for A6000
torch.set_float32_matmul_precision('high')

class FullEvaluator:
    @staticmethod
    def compute_metrics(batch, y_hat):
        """Calculates Realism, Physics, and Social metrics."""
        # 1. Best Mode Selection (MinFDE)
        reg_mask = ~batch['padding_mask'][:, 20:]
        l2_norm = (torch.norm(y_hat[:, :, :, :2] - batch.y, p=2, dim=-1) * reg_mask).sum(dim=-1)
        best_mode_idx = l2_norm.argmin(dim=0)
        best_trajs = y_hat[best_mode_idx, torch.arange(batch.num_nodes), :, :2]

        # 2. Heading Error (Rad)
        agent_idx = batch.agent_index
        v_pred = (best_trajs[agent_idx, -1] - best_trajs[agent_idx, -2])
        v_gt = (batch.y[agent_idx, -1] - batch.y[agent_idx, -2])
        h_err = torch.abs(torch.atan2(v_pred[:, 1], v_pred[:, 0]) - torch.atan2(v_gt[:, 1], v_gt[:, 0]))
        h_err = torch.where(h_err > np.pi, 2*np.pi - h_err, h_err)

        # 3. Jerk (m/s^3 approximation)
        # Using the primary agent's best trajectory
        traj = best_trajs[agent_idx] # [Batch, 30, 2]
        vel = traj[:, 1:] - traj[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]
        jerk = torch.norm(acc[:, 1:] - acc[:, :-1], p=2, dim=-1).mean()

        # 4. Proximity (Social)
        # Calculate the minimum distance between any two agents at t=final
        dists = torch.cdist(best_trajs[:, -1, :], best_trajs[:, -1, :])
        dists += torch.eye(best_trajs.size(0), device=best_trajs.device) * 100.0
        avg_min_sep = dists.min(dim=1)[0].mean() # Average distance to nearest neighbor

        return h_err.mean().item(), jerk.item(), avg_min_sep.item()

def main():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="/mount/arbeitsdaten/studenten4/rasoulta")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    model = HiVT.load_from_checkpoint(args.ckpt_path, strict=False).eval().cuda()
    dm = NuScenesHiVTDataModule(root=args.root, val_batch_size=args.batch_size)
    dm.setup(stage="validate")
    
    # Accumulators
    res = {"ade": [], "fde": [], "mr": [], "head": [], "jerk": [], "sep": []}

    print(f"--- Evaluating Checkpoint: {os.path.basename(args.ckpt_path)} ---")
    with torch.no_grad():
        for batch in tqdm(dm.val_dataloader()):
            batch = batch.to(model.device)
            y_hat, _ = model(batch)
            
            # Accuracy (Primary Agent)
            agent_idx = batch.agent_index
            y_agent = batch.y[agent_idx]
            y_hat_agent = y_hat[:, agent_idx, :, :2]
            fde_all = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
            best_m = fde_all.argmin(dim=0)
            y_best = y_hat_agent[best_m, torch.arange(len(agent_idx))]
            
            res["ade"].append(torch.norm(y_best - y_agent, p=2, dim=-1).mean().item())
            res["fde"].append(torch.norm(y_best[:, -1] - y_agent[:, -1], p=2, dim=-1).mean().item())
            res["mr"].append((torch.norm(y_best[:, -1] - y_agent[:, -1], p=2, dim=-1) > 2.0).float().mean().item())
            
            # Realism
            h, j, s = FullEvaluator.compute_metrics(batch, y_hat)
            res["head"].append(h)
            res["jerk"].append(j)
            res["sep"].append(s)

    print("\n" + "="*55)
    print(f"{'METRIC':<25} | {'VALUE':<15}")
    print("-" * 55)
    print(f"{'MinADE (Accuracy)':<25} | {np.mean(res['ade']):.4f} m")
    print(f"{'MinFDE (Accuracy)':<25} | {np.mean(res['fde']):.4f} m")
    print(f"{'Miss Rate':<25} | {np.mean(res['mr'])*100:.2f} %")
    print("-" * 55)
    print(f"{'Heading Error':<25} | {np.mean(res['head']):.4f} rad")
    print(f"{'Heading Error (Deg)':<25} | {np.degrees(np.mean(res['head'])):.2f} °")
    print(f"{'Average Jerk':<25} | {np.mean(res['jerk']):.6f}")
    print(f"{'Min Separation (Social)':<25} | {np.mean(res['sep']):.4f} m")
    print("="*55)

if __name__ == "__main__":
    main()