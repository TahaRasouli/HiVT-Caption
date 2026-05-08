import os
import glob
import torch
import numpy as np
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.cvae import CVAE
from models.cvae_gan import CVAE_GAN
from models.hivt import HiVT

torch.set_float32_matmul_precision('high')

CVAE_SAMPLES = 6  # stochastic draws from CVAE prior

# ──────────────────────────────────────────────────────────────────────────────
# Model type detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_model_type(run_dir):
    """
    Returns one of: 'cvae', 'hivt', or None (skip).
    Decision is based on hparams.yaml found in run_dir or its parent.
    """
    hparams_path = os.path.join(run_dir, "hparams.yaml")
    if not os.path.isfile(hparams_path):
        # Try one level up (e.g. run_dir is the 'checkpoints' sub-folder)
        hparams_path = os.path.join(os.path.dirname(run_dir), "hparams.yaml")
    if not os.path.isfile(hparams_path):
        return None

    with open(hparams_path) as f:
        hp = yaml.safe_load(f)

    # Skip maneuver-classification and captioning runs
    if hp.get("train_caption"):
        return None

    if hp.get("train_cvae"):
        return "cvae"

    if hp.get("train_cvae_gan"):
        return "cvae_gan"

    return "hivt"


# ──────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ──────────────────────────────────────────────────────────────────────────────

def predict_cvae(model, data, k=CVAE_SAMPLES):
    """Sample k trajectories from the CVAE prior → [k, N, 30, 2]."""
    context = model(data).reshape(-1, model.hparams.embed_dim)
    B = context.size(0)
    ctx_exp = context.repeat_interleave(k, dim=0)
    y_flat, _ = model.decoder(ctx_exp, y_gt=None)
    return y_flat.reshape(B, k, 30, 2).permute(1, 0, 2, 3)


def predict_hivt(model, data):
    """Run HiVT forward → [num_modes, N, 30, 2]."""
    y_hat, _ = model(data)
    return y_hat[:, :, :, :2]


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_batch_metrics(y_hat, data):
    """
    y_hat : [Modes, N, 30, 2]

    minADE/minFDE/MR are averaged over ALL agents (matching training validation_step).
    Physics metrics (heading, jerk, separation) are computed on the primary agent.
    """
    # ── accuracy metrics over ALL agents (matches training val) ──────────────
    y_gt = data.y                                        # [N, 30, 2]
    dist = torch.norm(y_hat - y_gt, p=2, dim=-1)        # [M, N, 30]

    min_ade, _ = dist.mean(dim=-1).min(dim=0)            # [N]
    min_fde, _ = dist[:, :, -1].min(dim=0)               # [N]

    ade = min_ade.mean()
    fde = min_fde.mean()
    mr  = (min_fde > 2.0).float().mean()

    # ── physics metrics on the primary agent ─────────────────────────────────
    agent_idx = data.agent_index
    best_all  = dist.mean(dim=-1).argmin(dim=0)           # [N]
    best_traj = y_hat[best_all, torch.arange(data.num_nodes)]  # [N, 30, 2]

    traj_a = best_traj[agent_idx]
    y_gt_a = y_gt[agent_idx]

    # Heading error
    v_pred = traj_a[:, -1] - traj_a[:, -2]
    v_gt   = y_gt_a[:, -1] - y_gt_a[:, -2]
    h_err  = torch.abs(torch.atan2(v_pred[:, 1], v_pred[:, 0]) -
                       torch.atan2(v_gt[:, 1],   v_gt[:, 0]))
    h_err  = torch.where(h_err > np.pi, 2 * np.pi - h_err, h_err).mean()

    # Jerk
    vel  = traj_a[:, 1:] - traj_a[:, :-1]
    acc  = vel[:, 1:] - vel[:, :-1]
    jerk = torch.norm(acc[:, 1:] - acc[:, :-1], p=2, dim=-1).mean()

    # Min pairwise separation at final step
    dists = torch.cdist(best_traj[:, -1], best_traj[:, -1])
    dists += torch.eye(best_traj.size(0), device=best_traj.device) * 1e6
    sep   = dists.min(dim=1)[0].mean()

    return {"ade": ade.item(), "fde": fde.item(), "mr": mr.item(),
            "head": h_err.item(), "jerk": jerk.item(), "sep": sep.item()}


# ──────────────────────────────────────────────────────────────────────────────
# Single-checkpoint evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(ckpt_path, model_type, val_loader, device):
    print(f"  Loading {os.path.basename(ckpt_path)} ...")
    try:
        if model_type == "cvae":
            model = CVAE.load_from_checkpoint(ckpt_path, strict=False)
        elif model_type == "cvae_gan":
            model = CVAE_GAN.load_from_checkpoint(ckpt_path, strict=False)
        else:
            model = HiVT.load_from_checkpoint(ckpt_path, strict=False)
    except Exception as e:
        print(f"  SKIP – could not load: {e}")
        return None

    model.eval().to(device)

    accum = {k: [] for k in ("ade", "fde", "mr", "head", "jerk", "sep")}

    with torch.no_grad():
        for batch in tqdm(val_loader, leave=False, desc="    batches"):
            batch = batch.to(device)
            try:
                if model_type in ("cvae", "cvae_gan"):
                    y_hat = predict_cvae(model, batch)
                else:
                    y_hat = predict_hivt(model, batch)
                metrics = compute_batch_metrics(y_hat, batch)
                for k, v in metrics.items():
                    accum[k].append(v)
            except Exception as e:
                print(f"    batch error: {e}")
                continue

    if not accum["ade"]:
        return None

    return {k: float(np.mean(v)) for k, v in accum.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────────────────────

def print_run_table(run_name, results):
    """results: list of (ckpt_name, metrics_dict)"""
    W = 30
    header = f"{'Checkpoint':<{W}} | {'minADE':>8} | {'minFDE':>8} | {'MR%':>7} | {'Heading°':>10} | {'Jerk':>10} | {'Sep(m)':>8}"
    sep    = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print(f"  RUN: {run_name}")
    print(sep)
    print(header)
    print(sep)
    for name, m in results:
        if m is None:
            print(f"  {name:<{W-2}} | {'FAILED':>8}")
            continue
        print(
            f"  {name:<{W-2}} | {m['ade']:>8.4f} | {m['fde']:>8.4f} | "
            f"{m['mr']*100:>7.2f} | {np.degrees(m['head']):>10.2f} | "
            f"{m['jerk']:>10.6f} | {m['sep']:>8.4f}"
        )
    print('='*len(header))


def print_summary(all_best):
    """all_best: list of (run_name, best_metrics_dict | None)"""
    W = 28
    print(f"\n\n{'#'*80}")
    print("  SUMMARY  –  best checkpoint per run")
    print('#'*80)
    header = (f"{'Run':<{W}} | {'minADE':>8} | {'minFDE':>8} | {'MR%':>7} | "
              f"{'Heading°':>10} | {'Jerk':>10} | {'Sep(m)':>8}")
    print(header)
    print('-' * len(header))
    for run_name, m in all_best:
        if m is None:
            print(f"  {run_name:<{W-2}} | {'SKIPPED / FAILED':>8}")
            continue
        print(
            f"  {run_name:<{W-2}} | {m['ade']:>8.4f} | {m['fde']:>8.4f} | "
            f"{m['mr']*100:>7.2f} | {np.degrees(m['head']):>10.2f} | "
            f"{m['jerk']:>10.6f} | {m['sep']:>8.4f}"
        )
    print('#'*80)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser()
    parser.add_argument("--root",      type=str, default="/mount/studenten/projects/rasoulta/dataset/")
    parser.add_argument("--ckpt_root", type=str, default="/mount/studenten/projects/rasoulta/checkpoints/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--run",        type=str, default=None,
                        help="Evaluate only this run sub-directory (default: all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build val loader once – shared across all checkpoints
    print("Setting up validation dataloader ...")
    dm = NuScenesHiVTDataModule(root=args.root, val_batch_size=args.batch_size)
    dm.setup(stage="validate")
    val_loader = dm.val_dataloader()

    # Discover run directories
    run_dirs = sorted(
        d for d in glob.glob(os.path.join(args.ckpt_root, "*"))
        if os.path.isdir(d)
    )
    if args.run:
        run_dirs = [d for d in run_dirs if os.path.basename(d) == args.run]

    summary = []  # (run_name, best_metrics)

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)

        model_type = detect_model_type(run_dir)
        if model_type is None:
            print(f"\n[SKIP] {run_name}  (maneuver-classifier or caption model)")
            summary.append((run_name, None))
            continue

        # Collect .ckpt files – prefer a 'checkpoints/' sub-directory if present
        ckpt_subdir = os.path.join(run_dir, "checkpoints")
        search_dir  = ckpt_subdir if os.path.isdir(ckpt_subdir) else run_dir
        ckpt_files  = sorted(glob.glob(os.path.join(search_dir, "*.ckpt")))

        if not ckpt_files:
            print(f"\n[SKIP] {run_name}  (no .ckpt files found)")
            summary.append((run_name, None))
            continue

        print(f"\n[RUN] {run_name}  ({model_type.upper()}, {len(ckpt_files)} checkpoints)")

        run_results = []
        best_metrics, best_fde = None, float("inf")

        for ckpt_path in ckpt_files:
            ckpt_name = os.path.basename(ckpt_path)
            metrics   = evaluate_checkpoint(ckpt_path, model_type, val_loader, device)
            run_results.append((ckpt_name, metrics))
            if metrics is not None and metrics["fde"] < best_fde:
                best_fde     = metrics["fde"]
                best_metrics = metrics

        print_run_table(run_name, run_results)
        summary.append((run_name, best_metrics))

    print_summary(summary)


if __name__ == "__main__":
    main()
