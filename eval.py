import os
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.hivt import HiVT

# Performance optimization for A6000
torch.set_float32_matmul_precision('high')

class FastRealismEvaluator:
    def __init__(self, raster_dir):
        self.masks = {}
        self.meta = {}
        for city in ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']:
            mask_path = os.path.join(raster_dir, f"{city}_mask.npy")
            meta_path = os.path.join(raster_dir, f"{city}_meta.npy")
            if os.path.exists(mask_path):
                self.masks[city] = np.load(mask_path)
                self.meta[city] = np.load(meta_path) # [edge_x, edge_y, res]

    def is_off_road(self, pt, city):
        if city not in self.masks: return False
        
        # 1. Retrieve mask and metadata
        mask = self.masks[city]
        width, height, res = self.meta[city]
        
        # 2. NuScenes map offset handling
        # IMPORTANT: Global NuScenes maps have a specific minimum (x, y)
        # For 'singapore-onenorth', min_x is 0, but for others, it varies.
        # However, the nmap.get_map_mask usually centers the patch box.
        # Since we used patch_box = (width/2, height/2, height, width),
        # pixel (0,0) in the mask corresponds to global (0,0).
        
        px = int(pt[0] / res)
        py = int(pt[1] / res)
        
        # 3. Boundary and Orientation Check
        # NuScenes Bitmaps usually follow (Row=Y, Col=X)
        # We also need to flip the Y axis because images start from Top-Left (0,0)
        # while maps start from Bottom-Left (0,0)
        
        if 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1]:
            # Flip Y to account for image vs cartesian coordinates
            flipped_py = mask.shape[0] - 1 - py
            return mask[flipped_py, px] == 0 
            
        return True # Treat out-of-bounds as off-road

    def compute_metrics(self, batch, y_hat):
        # 1. Select best trajectory per agent (MinFDE)
        reg_mask = ~batch['padding_mask'][:, 20:]
        l2_norm = (torch.norm(y_hat[:, :, :, :2] - batch.y, p=2, dim=-1) * reg_mask).sum(dim=-1)
        best_mode_idx = l2_norm.argmin(dim=0)
        best_trajs = y_hat[best_mode_idx, torch.arange(batch.num_nodes), :, :2]

        # 2. Heading Error (Rad) - Focus on primary agents
        agent_idx = batch.agent_index
        v_pred = (best_trajs[agent_idx, -1] - best_trajs[agent_idx, -2])
        v_gt = (batch.y[agent_idx, -1] - batch.y[agent_idx, -2])
        heading_err = torch.abs(torch.atan2(v_pred[:, 1], v_pred[:, 0]) - 
                                torch.atan2(v_gt[:, 1], v_gt[:, 0]))
        heading_err = torch.where(heading_err > np.pi, 2*np.pi - heading_err, heading_err)

        # 3. Fast Off-Road Rate
        off_road_count = 0
        origin = batch.origin.cpu().numpy()
        theta = batch.theta.cpu().numpy()
        
        for i, idx in enumerate(agent_idx):
            city = batch.city[i] if hasattr(batch, 'city') else 'singapore-onenorth'
            local_dest = best_trajs[idx, -1].cpu().numpy()
            c, s = np.cos(-theta[i]), np.sin(-theta[i])
            rot_mat = np.array([[c, -s], [s, c]])
            global_dest = (local_dest @ rot_mat.T) + origin[i]
            
            if self.is_off_road(global_dest, city):
                off_road_count += 1

        # 4. Collision Rate (Spatial check)
        collision = 0
        for t in [14, 29]: # Check at 1.5s and 3.0s
            dists = torch.cdist(best_trajs[:, t, :], best_trajs[:, t, :])
            dists += torch.eye(best_trajs.size(0), device=best_trajs.device) * 10.0
            if (dists < 1.8).any(): 
                collision = 1
                break

        return heading_err.mean().item(), off_road_count / len(agent_idx), collision

def main():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="/mount/arbeitsdaten/studenten4/rasoulta")
    parser.add_argument("--nusc_root", type=str, default="/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta/v1.0-trainval/")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    model = HiVT.load_from_checkpoint(args.ckpt_path, strict=False).eval()
    evaluator = FastRealismEvaluator(os.path.join(args.nusc_root, "raster_maps"))
    
    dm = NuScenesHiVTDataModule(root=args.root, val_batch_size=args.batch_size)
    dm.setup(stage="validate")
    
    h_errs, or_rates, colls = [], [], []
    
    for batch in tqdm(dm.val_dataloader(), desc="Evaluating"):
        batch = batch.to(model.device)
        with torch.no_grad():
            y_hat, _ = model(batch)
            h, o, c = evaluator.compute_metrics(batch, y_hat)
            h_errs.append(h); or_rates.append(o); colls.append(c)

    print(f"\nHEADING ERROR: {np.mean(h_errs):.4f} rad")
    print(f"OFF-ROAD RATE: {np.mean(or_rates)*100:.2f}%")
    print(f"COLLISION RATE: {np.mean(colls)*100:.2f}%")

if __name__ == "__main__":
    main()