import os
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.hivt import HiVT
from models.cvae import CVAE

torch.set_float32_matmul_precision('high')

CVAE_SAMPLES = 6


# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────

def predict(model, model_type, data):
    """Returns y_hat [Modes, N, 30, 2] regardless of model type."""
    if model_type == "hivt":
        y_hat, _ = model(data)
        return y_hat[:, :, :, :2]

    # CVAE / CVAE_GAN: sample K trajectories from prior
    context = model(data).reshape(-1, model.hparams.embed_dim)
    B = context.size(0)
    ctx_exp = context.repeat_interleave(CVAE_SAMPLES, dim=0)
    y_flat, _ = model.decoder(ctx_exp, y_gt=None)
    return y_flat.reshape(B, CVAE_SAMPLES, 30, 2).permute(1, 0, 2, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_hat, data):
    """
    y_hat : [Modes, N, 30, 2]
    minADE/minFDE/MR over ALL agents (matches training validation_step).
    Physics metrics on primary agent only.
    """
    y_gt = data.y                                      # [N, 30, 2]
    dist = torch.norm(y_hat - y_gt, p=2, dim=-1)      # [M, N, 30]

    min_ade, _ = dist.mean(dim=-1).min(dim=0)          # [N]
    min_fde, _ = dist[:, :, -1].min(dim=0)             # [N]

    ade = min_ade.mean().item()
    fde = min_fde.mean().item()
    mr  = (min_fde > 2.0).float().mean().item()

    # Physics on primary agent
    agent_idx = data.agent_index
    best_all  = dist.mean(dim=-1).argmin(dim=0)
    best_traj = y_hat[best_all, torch.arange(data.num_nodes)]  # [N, 30, 2]

    traj_a = best_traj[agent_idx]
    y_gt_a = y_gt[agent_idx]

    v_pred = traj_a[:, -1] - traj_a[:, -2]
    v_gt   = y_gt_a[:, -1] - y_gt_a[:, -2]
    h_err  = torch.abs(torch.atan2(v_pred[:, 1], v_pred[:, 0]) -
                       torch.atan2(v_gt[:, 1],   v_gt[:, 0]))
    h_err  = torch.where(h_err > np.pi, 2 * np.pi - h_err, h_err).mean().item()

    vel  = traj_a[:, 1:] - traj_a[:, :-1]
    acc  = vel[:, 1:] - vel[:, :-1]
    jerk = torch.norm(acc[:, 1:] - acc[:, :-1], p=2, dim=-1).mean().item()

    dists = torch.cdist(best_traj[:, -1], best_traj[:, -1])
    dists += torch.eye(best_traj.size(0), device=best_traj.device) * 1e6
    sep   = dists.min(dim=1)[0].mean().item()

    return {"ade": ade, "fde": fde, "mr": mr, "head": h_err, "jerk": jerk, "sep": sep}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["hivt", "cvae", "cvae_gan"],
                        help="Model architecture to evaluate")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to the .ckpt file")
    parser.add_argument("--root", type=str, default="/mount/studenten/projects/rasoulta/dataset/")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Load model
    if args.model == "cvae_gan":
        from models.cvae_gan import CVAE_GAN
        cls = CVAE_GAN
    else:
        cls = {"hivt": HiVT, "cvae": CVAE}[args.model]
    print(f"Loading {args.model.upper()} from {os.path.basename(args.ckpt_path)} ...")
    model = cls.load_from_checkpoint(args.ckpt_path, strict=False).eval().cuda()

    # Data
    dm = NuScenesHiVTDataModule(root=args.root, val_batch_size=args.batch_size)
    dm.setup(stage="validate")

    accum = {k: [] for k in ("ade", "fde", "mr", "head", "jerk", "sep")}

    with torch.no_grad():
        for batch in tqdm(dm.val_dataloader()):
            batch = batch.to(model.device)
            y_hat = predict(model, args.model, batch)
            m = compute_metrics(y_hat, batch)
            for k, v in m.items():
                accum[k].append(v)

    print("\n" + "=" * 55)
    print(f"{'METRIC':<25} | {'VALUE':<15}")
    print("-" * 55)
    print(f"{'MinADE':<25} | {np.mean(accum['ade']):.4f} m")
    print(f"{'MinFDE':<25} | {np.mean(accum['fde']):.4f} m")
    print(f"{'Miss Rate (>2 m)':<25} | {np.mean(accum['mr']) * 100:.2f} %")
    print("-" * 55)
    print(f"{'Heading Error':<25} | {np.degrees(np.mean(accum['head'])):.2f} °")
    print(f"{'Jerk':<25} | {np.mean(accum['jerk']):.6f}")
    print(f"{'Min Separation':<25} | {np.mean(accum['sep']):.4f} m")
    print("=" * 55)


if __name__ == "__main__":
    main()
