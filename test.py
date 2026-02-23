import os
import json
import torch
import numpy as np

# --- SERVER-SIDE FIX ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# -----------------------

from matplotlib.patches import Polygon as MplPolygon
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from models.cvae import CVAE
from shapely.geometry import Polygon as ShapelyPolygon

# =========================================================
# CONFIGURATION
# =========================================================
NUSCENES_DATAROOT = "/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta/"
DATA_DIR = "/mount/studenten/projects/rasoulta/dataset/train_processed"
CHECKPOINT = "/mount/studenten/projects/rasoulta/checkpoints/vae_best/checkpoints/epoch=47-step=15888.ckpt"
# Use absolute path or clear relative path
OUT_DIR = os.path.join(os.getcwd(), "inference_diverse_10_samples")
NUSCENES_VERSION = "v1.0-trainval"

TARGET_SAMPLES = 10       
K = 6                     
DT = 0.1                  
ACTOR_FILTER_DIST = 15.0  
STATIONARY_THRESHOLD = 0.5 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
map_cache = {}
nusc_cache = None

os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# DATABASE LOADERS & UTILS
# =========================================================

def get_nusc():
    global nusc_cache
    if nusc_cache is None:
        nusc_cache = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)
    return nusc_cache

def get_map(location):
    if location not in map_cache:
        map_cache[location] = NuScenesMap(dataroot=NUSCENES_DATAROOT, map_name=location)
    return map_cache[location]

def local_to_global(coords_local, origin, theta):
    if isinstance(coords_local, torch.Tensor): coords_local = coords_local.cpu().numpy()
    cos, sin = np.cos(theta), np.sin(theta)
    rot_inv = np.array([[cos, -sin], [sin, cos]]) 
    if coords_local.ndim == 1: coords_local = coords_local.reshape(1, 2)
    return (coords_local @ rot_inv.T) + origin

def get_yaw(vector): return np.arctan2(vector[1], vector[0])

def get_yaw_change(yaw_start, yaw_end):
    return np.degrees(np.unwrap([yaw_start, yaw_end])[1] - yaw_start)

# =========================================================
# SCENE ANALYZER
# =========================================================

class SceneAnalyzer:
    def __init__(self, nmap, nusc):
        self.nmap = nmap
        self.nusc = nusc

    def get_lane_record(self, token):
        try: return self.nmap.get('lane', token)
        except KeyError:
            try: return self.nmap.get('lane_connector', token)
            except KeyError: return None

    def get_heading(self, lane_token):
        rec = self.get_lane_record(lane_token)
        if not rec: return 0.0
        try:
            n1 = self.nmap.get('node', rec['exterior_node_tokens'][0])
            n2 = self.nmap.get('node', rec['exterior_node_tokens'][-1])
            return np.arctan2(n2['y'] - n1['y'], n2['x'] - n1['x'])
        except: return 0.0

    def get_structured_lanes(self, ego_pos):
        ego_lane = self.nmap.get_closest_lane(ego_pos[0], ego_pos[1], radius=2.5)
        if not ego_lane: return None, [], None
        
        block_token = None
        for rb in self.nmap.road_block:
            if ego_lane in rb.get('lane_tokens', []):
                block_token = rb['token']
                break

        ego_heading = self.get_heading(ego_lane)
        patch = [ego_pos[0]-15, ego_pos[1]-15, ego_pos[0]+15, ego_pos[1]+15]
        try:
            nearby = self.nmap.get_records_in_patch(patch, ['lane'], mode='intersect')['lane']
        except: nearby = [ego_lane]

        parallel = []
        for l in nearby:
            h = self.get_heading(l)
            if abs(np.arctan2(np.sin(h-ego_heading), np.cos(h-ego_heading))) < 0.45:
                parallel.append(l)
        
        parallel = list(set(parallel))
        perp_x, perp_y = -np.sin(ego_heading), np.cos(ego_heading)
        scores = []
        for l in parallel:
            rec = self.get_lane_record(l)
            if not rec: continue
            node = self.nmap.get('node', rec['exterior_node_tokens'][0])
            scores.append((node['x']*perp_x + node['y']*perp_y, l))
        scores.sort(key=lambda x: x[0])
        return ego_lane, [s[1] for s in scores], block_token

# =========================================================
# VISUALIZATION
# =========================================================

def visualize_sample(data, candidates_global, nmap, filename):
    origin = [float(x) for x in data.origin[0].cpu().numpy()]
    theta = float(data.theta.cpu().item())
    pr = 40
    patch_box = (origin[0]-pr, origin[1]-pr, origin[0]+pr, origin[1]+pr)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#111111')

    layers = {'drivable_area': '#2c2c2c', 'lane': '#1a2a3a', 'stop_line': '#bf9b30'}
    for layer, color in layers.items():
        try:
            tokens = nmap.get_records_in_patch(patch_box, [layer], mode='intersect').get(layer, [])
            for t in tokens:
                rec = nmap.get(layer, t)
                poly = nmap.extract_polygon(rec.get('polygon_token', t))
                if not poly.is_empty:
                    ax.add_patch(MplPolygon(np.array(poly.exterior.coords)[:, :2], closed=True, facecolor=color, alpha=0.6, zorder=1))
        except: continue

    ego_hist = local_to_global(data.positions[data.av_index, :20].cpu().numpy(), origin, theta)
    ax.plot(ego_hist[:, 0], ego_hist[:, 1], color='#00A2FF', linewidth=3, linestyle='--', zorder=16)
    
    colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, K))
    for k in range(K):
        traj = candidates_global[k]
        ax.plot(traj[:, 0], traj[:, 1], color=colors[k], linewidth=3.5, zorder=18)
        ax.text(traj[-1, 0], traj[-1, 1], f"M{k}", color='white', fontsize=10, weight='bold', zorder=21, bbox=dict(facecolor=colors[k], alpha=0.8, edgecolor='none', pad=1))
        ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[k], s=80, marker='X', zorder=19)

    ax.scatter(origin[0], origin[1], color='#00A2FF', s=180, edgecolors='white', zorder=20)
    ax.set_xlim(origin[0]-pr, origin[0]+pr); ax.set_ylim(origin[1]-pr, origin[1]+pr); ax.axis('off')
    
    plt.savefig(os.path.join(OUT_DIR, f"{filename}.png"), bbox_inches='tight', dpi=120, facecolor='#111111')
    plt.close(fig)

# =========================================================
# MAIN
# =========================================================

def main():
    print(f"Checking output directory: {OUT_DIR}")
    nusc = get_nusc()
    model = CVAE.load_from_checkpoint(CHECKPOINT, map_location=device).to(device).eval()
    all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.pt')])
    
    final_json_data = []
    processed = 0

    for fname in all_files:
        if processed >= TARGET_SAMPLES: break
        try:
            data = torch.load(os.path.join(DATA_DIR, fname), map_location=device)
            nmap = get_map(data.city); analyzer = SceneAnalyzer(nmap, nusc)
            origin = data.origin[0].cpu().numpy(); theta = data.theta.cpu().item()
            
            ego_lane, sorted_lanes, ego_block = analyzer.get_structured_lanes(origin)
            
            # LOOSENED FILTER: Even 2 lanes are okay if needed, but 3 is better.
            if not ego_lane or len(sorted_lanes) < 2: continue
            
            with torch.no_grad():
                context = model(data).reshape(-1, model.hparams.embed_dim)
                ego_context = context[int(data.av_index)].repeat(K, 1)
                traj_local, _ = model.decoder(ego_context, None)
                traj_local = traj_local.reshape(K, 30, 2).detach().cpu().numpy()
            
            candidates_global = [local_to_global(traj_local[k], origin, theta) for k in range(K)]

            modes_metadata = []
            distinct_intents = set()
            
            for k in range(K):
                t_glob = candidates_global[k]
                t_loc = traj_local[k]
                end_pos = t_glob[-1]
                end_lane = nmap.get_closest_lane(end_pos[0], end_pos[1], radius=2.0)
                
                if end_lane == ego_lane: lat_stat = "keep_lane"
                elif end_lane in sorted_lanes:
                    lat_stat = "lane_change_left" if sorted_lanes.index(end_lane) > sorted_lanes.index(ego_lane) else "lane_change_right"
                else:
                    outgoing = nmap.get_outgoing_lane_ids(ego_lane)
                    is_succ = any(end_lane == out or end_lane in nmap.get_outgoing_lane_ids(out) for out in outgoing)
                    lat_stat = "keep_lane_crossing_intersection" if is_succ else "turning/exit"

                prog = []
                for i in range(3):
                    idx_s, idx_e = i*10, (i+1)*10-1
                    dist_covered = np.linalg.norm(t_loc[idx_e] - t_loc[idx_s])
                    if dist_covered < STATIONARY_THRESHOLD:
                        m_type = "stationary"; yaw_d = 0.0
                    else:
                        v1 = t_glob[idx_s+1]-t_glob[idx_s] if idx_s+1<30 else t_glob[idx_s]-t_glob[idx_s-1]
                        v2 = t_glob[idx_e]-t_glob[idx_e-1]
                        yaw_d = float(get_yaw_change(get_yaw(v1), get_yaw(v2)))
                        m_type = "straight" if abs(yaw_d) < 5 else "turn"
                    prog.append({"sec": i+1, "maneuver": m_type})

                # Intent is a combination of lateral status and maneuver
                distinct_intents.add((lat_stat, tuple(p['maneuver'] for p in prog)))
                modes_metadata.append({"id": k, "status": lat_stat, "progression": prog})

            # DIVERSITY CHECK
            if len(distinct_intents) < 2: continue

            processed += 1
            sample_result = {
                "file": fname,
                "ego_state": {"lane_idx": sorted_lanes.index(ego_lane)+1, "total_lanes": len(sorted_lanes)},
                "predictions": modes_metadata
            }
            final_json_data.append(sample_result)
            
            print(f"SUCCESS: Found and processed {fname} ({processed}/10)")
            visualize_sample(data, candidates_global, nmap, fname)

        except Exception as e:
            continue

    # Save JSON to the same folder
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(final_json_data, f, indent=2)
    print(f"All done. Results saved in {OUT_DIR}")

if __name__ == "__main__":
    main()