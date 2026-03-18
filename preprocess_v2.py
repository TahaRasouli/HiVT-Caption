import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyquaternion import Quaternion
from collections import deque
from scipy.interpolate import interp1d
import gc
import random

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

# ================= CONFIGURATION =================
DATAROOT = "/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta/"
VERSION = "v1.0-trainval"

IMAGE_DIR = "/mount/studenten/projects/rasoulta/dataset/caption-visuals"
PT_DIR = "/mount/studenten/projects/rasoulta/dataset/train_processed_labeled"

TARGET_SAMPLE_COUNT = 5000  # High limit, safety break will handle the 1000 png cap
MAX_PNG_COUNT = 1000
FORWARD_MAX, BACKWARD_MAX, LATERAL_LIMIT = 50.0, 10.0, 25.0

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PT_DIR, exist_ok=True)

# ================= GEOMETRY HELPERS =================

def get_yaw(q):
    return Quaternion(q).yaw_pitch_roll[0]

def global_to_local(points_global, ego_translation, ego_rotation):
    pts = np.array(points_global)[:, :2] - np.array(ego_translation[:2])
    yaw = get_yaw(ego_rotation)
    c, s = np.cos(-yaw), np.sin(-yaw)
    rot = np.array([[c, -s], [s, c]])
    local = np.dot(pts, rot.T)
    # x is lateral, y is forward
    return np.stack([-local[:, 1], local[:, 0]], axis=1)

def crop_forward_sector(local_coords):
    if len(local_coords) == 0: return local_coords
    x, y = local_coords[:, 0], local_coords[:, 1]
    mask = ((y <= FORWARD_MAX) & (y >= -BACKWARD_MAX) & (np.abs(x) <= LATERAL_LIMIT))
    return local_coords[mask]

# ================= ROUTING ENGINE =================

def get_batches(nmap, inf_map, ego_trans):
    legal = set(inf_map.keys())
    dist_map = {t: np.linalg.norm(inf_map[t][-1] - inf_map[t][0]) for t in legal}
    
    # Identify current lane and forbidden predecessors
    ego_lane_id = nmap.get_closest_lane(ego_trans[0], ego_trans[1], radius=3.0)
    forbidden_ids = set()
    if ego_lane_id:
        try:
            rec = nmap.get('lane', ego_lane_id)
        except:
            rec = nmap.get('lane_connector', ego_lane_id)
        if rec and 'predecessor' in rec:
            forbidden_ids.update(rec['predecessor'])

    # Root selection: proximity + forward-facing + not a predecessor
    raw_roots = [t for t in legal if np.min(np.linalg.norm(inf_map[t], axis=1)) <= 5.0]
    roots = []
    for r in raw_roots:
        if r in forbidden_ids: continue
        pts = inf_map[r]
        if pts[-1, 1] > 0 and pts[-1, 1] > pts[0, 1]:
            roots.append(r)
    roots = sorted(roots, key=lambda t: np.min(np.linalg.norm(inf_map[t], axis=1)))[:5]
    
    def walk(curr, chain, current_dist):
        new_dist = current_dist + dist_map.get(curr, 0)
        if new_dist >= FORWARD_MAX or len(chain) >= 15: 
            return [chain]
        res = []
        outgoing = [s for s in nmap.get_outgoing_lane_ids(curr) if s in legal and s not in chain]
        for s in outgoing:
            res.extend(walk(s, chain + [s], new_dist))
        return res if res else [chain]
    
    all_paths = []
    for r in roots:
        all_paths.extend(walk(r, [r], dist_map.get(r, 0)))
        
    all_paths.sort(key=len, reverse=True)
    unique_paths, seen_sets = [], []
    for p in all_paths:
        p_set = set(p)
        if not any(p_set.issubset(s) for s in seen_sets):
            unique_paths.append(p)
            seen_sets.append(p_set)
            
    batches = []
    for p in unique_paths:
        stitched = np.vstack([inf_map[t] for t in p])
        cropped = crop_forward_sector(stitched)
        if len(cropped) > 2 and cropped[-1, 1] > 5.0:
            batches.append((cropped, p))
    return batches

# ================= VISUALS & ENCODING =================

def save_production_visuals(batches, sample_token):
    for target_idx, (target_geom, _) in enumerate(batches):
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        ax.set_facecolor('black')
        ax.set_xlim(-LATERAL_LIMIT, LATERAL_LIMIT)
        ax.set_ylim(-BACKWARD_MAX, FORWARD_MAX)
        ax.axis('off')

        # 1. Alts (Green)
        for i, (path_geom, _) in enumerate(batches):
            if i != target_idx:
                ax.plot(path_geom[:, 0], path_geom[:, 1], color='#00FF00', lw=4, alpha=0.25, zorder=5)

        # 2. Target (Red)
        ax.plot(target_geom[:, 0], target_geom[:, 1], color='#FF0000', lw=6, zorder=10)
        ax.plot(0, 0, 'w^', markersize=12, zorder=100) 
        
        plt.savefig(os.path.join(IMAGE_DIR, f"{sample_token}_r{target_idx}.png"), facecolor='black', bbox_inches='tight', dpi=100)
        plt.close(fig)

def calculate_feature_matrix(local_points, target_points=50):
    diffs = np.diff(local_points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    _, idx = np.unique(cum_dist, return_index=True)
    if len(idx) < 3: return None
    pts, cum = local_points[idx], cum_dist[idx]
    if cum[-1] < 1.0: return None

    new_dists = np.linspace(0, cum[-1], target_points)
    interp_pts = interp1d(cum, pts, axis=0)(new_dists)
    g = np.gradient(interp_pts, axis=0)
    u = g / (np.linalg.norm(g, axis=1)[:, np.newaxis] + 1e-6)
    
    # 9-col matrix [x, y, ux, uy, kappa, s, zero, zero, zero]
    # (Simplified placeholders for intersection/connector to keep logic focused)
    matrix = np.zeros((target_points, 9))
    matrix[:, :2] = interp_pts
    matrix[:, 2:4] = u
    matrix[:, 5] = new_dists / cum[-1]
    return matrix

# ================= MAIN LOOP =================

def main():
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    all_tokens = [s['token'] for s in nusc.sample]
    random.shuffle(all_tokens)
    
    map_cache = {}
    processed_count = 0
    pbar = tqdm(total=TARGET_SAMPLE_COUNT, desc="🚀 Processing")
    
    for token in all_tokens:
        # SAFETY CHECK: Stop if we hit 1000 PNGs
        if processed_count % 100 == 0:
            current_pngs = len([f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')])
            if current_pngs >= MAX_PNG_COUNT:
                print(f"\n[STOP] Reached {current_pngs} PNGs. Terminating.")
                break

        sample = nusc.get('sample', token)
        loc = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']
        if loc not in map_cache: map_cache[loc] = NuScenesMap(DATAROOT, loc)
        nmap = map_cache[loc]
        
        ego = nusc.get('ego_pose', nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token'])
        trans, rot = ego['translation'], ego['rotation']
        
        # Build local reachability map
        ego_lane = nmap.get_closest_lane(trans[0], trans[1], radius=5.0)
        if not ego_lane: continue
        
        queue, visited, node_paths = deque([ego_lane]), set(), {}
        while queue:
            t = queue.popleft()
            if t in visited: continue
            visited.add(t)
            try:
                pts = nmap.discretize_lanes([t], 0.5)[t]
                node_paths[t] = np.array(pts)[:, :2]
                for s in nmap.get_outgoing_lane_ids(t): queue.append(s)
            except: continue

        inf_map = {t: global_to_local(p, trans, rot) for t, p in node_paths.items()}
        batches = get_batches(nmap, inf_map, trans)
        
        if batches:
            success = False
            for b_idx, (path, _) in enumerate(batches):
                matrix = calculate_feature_matrix(path)
                if matrix is not None:
                    torch.save(torch.from_numpy(matrix).float(), os.path.join(PT_DIR, f"{token}_r{b_idx}.pt"))
                    success = True
            
            if success:
                save_production_visuals(batches, token)
                processed_count += 1
                pbar.update(1)
            
        if processed_count % 20 == 0: gc.collect()

if __name__ == "__main__":
    main()