
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Stable headless mode for high-volume processing
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyquaternion import Quaternion
from collections import deque
from scipy.interpolate import interp1d
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Point
from shapely.ops import unary_union
import gc
import random

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

# ================= CONFIGURATION =================
DATAROOT = "/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta/"
VERSION = "v1.0-trainval"

IMAGE_DIR = "/mount/studenten/projects/rasoulta/dataset/caption-visuals"
PT_DIR = "/mount/studenten/projects/rasoulta/dataset/train_processed_labeled"

TARGET_SAMPLE_COUNT = 500
FORWARD_MAX, BACKWARD_MAX, LATERAL_LIMIT = 50.0, 10.0, 25.0
FIXED_POINT_COUNT = 50 

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
    return np.stack([-local[:, 1], local[:, 0]], axis=1)

def crop_forward_sector(local_coords):
    if len(local_coords) == 0: return local_coords
    x, y = local_coords[:, 0], local_coords[:, 1]
    mask = ((y <= FORWARD_MAX) & (y >= -BACKWARD_MAX) & (np.abs(x) <= LATERAL_LIMIT))
    return local_coords[mask]

# ================= FEATURE & VISUAL LOGIC =================

def calculate_feature_matrix_9col(local_points, ids_in_batch, nmap, ego_trans, ego_rot, target_points=50):
    diffs = np.diff(local_points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    _, idx = np.unique(cum_dist, return_index=True)
    if len(idx) < 3: return None
    
    pts, cum = local_points[idx], cum_dist[idx]
    total_len = cum[-1]
    if total_len < 1.0: return None

    new_dists = np.linspace(0, total_len, target_points)
    interp_pts = interp1d(cum, pts, axis=0)(new_dists)
    
    g = np.gradient(interp_pts, axis=0)
    norm = np.linalg.norm(g, axis=1)[:, np.newaxis]
    norm[norm == 0] = 1.0
    u = g / norm
    g2 = np.gradient(g, axis=0)
    k = (g[:, 0]*g2[:, 1] - g[:, 1]*g2[:, 0]) / (np.power(norm[:, 0], 3) + 1e-6)
    
    yaw = get_yaw(ego_rot)
    c, s = np.cos(yaw), np.sin(yaw)
    rot_inv = np.array([[c, s], [-s, c]])
    remap_pts = np.stack([interp_pts[:, 1], -interp_pts[:, 0]], axis=1)
    pts_global = np.dot(remap_pts, rot_inv) + np.array(ego_trans[:2])
    
    is_int, is_conn = np.zeros(target_points), np.zeros(target_points)
    center = np.mean(pts_global, axis=0)
    radius = (total_len / 2) + 15
    
    try:
        nearby_road = nmap.get_records_in_radius(center[0], center[1], radius, ['road_segment'])['road_segment']
        junction_polys = [nmap.extract_polygon(nmap.get('road_segment', t)['polygon_token']) 
                         for t in nearby_road if nmap.get('road_segment', t).get('is_intersection', False)]
        junction_zone = unary_union(junction_polys) if junction_polys else None
        nearby_conn = set(nmap.get_records_in_radius(center[0], center[1], radius, ['lane_connector'])['lane_connector'])

        for i in range(target_points):
            p_shapely = Point(pts_global[i,0], pts_global[i,1])
            if junction_zone and junction_zone.contains(p_shapely):
                is_int[i] = 1.0
            closest_lane = nmap.get_closest_lane(pts_global[i,0], pts_global[i,1], radius=3.0)
            if closest_lane in nearby_conn:
                is_conn[i] = 1.0
    except: pass

    return np.stack([interp_pts[:, 0], interp_pts[:, 1], u[:, 0], u[:, 1], 
                    np.clip(k, -0.5, 0.5), new_dists/total_len, np.zeros(target_points),
                    is_int, is_conn], axis=1)

def save_batch_visuals_production(nmap, ego_trans, ego_rot, batches, inf_map, sample_token):
    for b_idx, (path_geom, batch_ids) in enumerate(batches):
        num_stages = len(batch_ids) - 1
        for stage in range(1, num_stages + 1):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor='black')
            plt.subplots_adjust(wspace=0.02)
            
            for ax in [ax1, ax2]:
                ax.set_facecolor('black')
                ax.set_xlim(-LATERAL_LIMIT, LATERAL_LIMIT)
                ax.set_ylim(-BACKWARD_MAX, FORWARD_MAX)
                ax.axis('off')
                ax.plot(0, 0, 'r^', markersize=14, zorder=100) # Ego position

            active_ids = [batch_ids[stage-1], batch_ids[stage]]
            
            # 1. Background paths
            for t_id, local_pts in inf_map.items():
                if t_id not in active_ids:
                    ax1.plot(local_pts[:, 0], local_pts[:, 1], color='#181818', lw=1.5, zorder=5)

            # 2. Foreground Highlight (High visibility)
            for t_id in active_ids:
                if t_id in inf_map:
                    local_pts = inf_map[t_id]
                    ax1.plot(local_pts[:, 0], local_pts[:, 1], color='#00FF00', lw=10, zorder=50)
                    m = len(local_pts) // 2
                    ax1.text(local_pts[m,0], local_pts[m,1], t_id[:4], color='white', 
                             weight='bold', zorder=60, bbox=dict(facecolor='black', edgecolor='#00FF00', pad=1))

            # 3. Context Map Rendering
            recs = nmap.get_records_in_radius(ego_trans[0], ego_trans[1], 60, ['lane', 'road_segment'])
            for lyr in ['lane', 'road_segment']:
                for t in recs.get(lyr, []):
                    rec = nmap.get(lyr, t)
                    if 'polygon_token' not in rec: continue
                    poly = nmap.extract_polygon(rec['polygon_token'])
                    coords = crop_forward_sector(global_to_local(np.array(poly.exterior.coords)[:,:2], ego_trans, ego_rot))
                    if len(coords) >= 3:
                        ax2.add_patch(MplPolygon(coords, facecolor='#0d0d0d', edgecolor='#1a1a1a', lw=0.5, zorder=1))

            # Overlay Titles
            ax1.text(-LATERAL_LIMIT + 2, FORWARD_MAX - 5, f"BATCH: {b_idx}", color='#00FF00', fontsize=22, weight='bold')
            ax1.text(-LATERAL_LIMIT + 2, FORWARD_MAX - 12, f"STAGE: {stage}", color='#00FF00', fontsize=22, weight='bold')
            ax1.text(-LATERAL_LIMIT + 2, FORWARD_MAX - 19, f"IDs: {active_ids[0][:4]} -> {active_ids[1][:4]}", color='white', fontsize=12)

            plt.savefig(os.path.join(IMAGE_DIR, f"{sample_token}_batch{b_idx}_stage{stage}.png"), facecolor='black', bbox_inches='tight', dpi=100)
            plt.close(fig)

# ================= ENGINES =================

def get_reachable_lanes(nmap, trans, rot):
    ego_lane = nmap.get_closest_lane(trans[0], trans[1], radius=5.0)
    if not ego_lane: return {}
    yaw = get_yaw(rot); ego_fwd = np.array([np.cos(yaw), np.sin(yaw)])
    queue, visited, node_path = deque([ego_lane]), set(), {}
    while queue:
        t = queue.popleft()
        if t in visited: continue
        visited.add(t)
        try:
            pts = nmap.discretize_lanes([t], 0.5)[t]
            node_path[t] = np.array(pts)[:, :2]
            for s in nmap.get_outgoing_lane_ids(t): queue.append(s)
            for adj in [nmap.get('lane', t).get('left_lane_token'), nmap.get('lane', t).get('right_lane_token')]:
                if adj: queue.append(adj)
        except: continue
    return node_path

def get_batches(nmap, inf_map):
    legal = set(inf_map.keys())
    roots = sorted([t for t in legal if np.min(np.linalg.norm(inf_map[t], axis=1)) <= 5.0], key=lambda t: np.min(np.linalg.norm(inf_map[t], axis=1)))[:5]
    def walk(curr, chain):
        if len(chain) >= 4 or inf_map[curr][-1, 1] >= FORWARD_MAX: return [chain]
        res = []
        for s in nmap.get_outgoing_lane_ids(curr):
            if s in legal and s not in chain: res.extend(walk(s, chain + [s]))
        return res if res else [chain]
    batches, seen = [], set()
    for r in roots:
        for p in walk(r, [r]):
            if tuple(p) not in seen:
                stitched = np.vstack([inf_map[t] for t in p])
                batches.append((crop_forward_sector(stitched), tuple(p))); seen.add(tuple(p))
    return batches

# ================= MAIN PRODUCTION LOOP =================

def main():
    print(f"[INIT] Loading NuScenes {VERSION}...")
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    
    # Collect every available sample token across all scenes
    all_tokens = []
    for scene in nusc.scene:
        curr = scene['first_sample_token']
        while curr:
            all_tokens.append(curr)
            curr = nusc.get('sample', curr)['next']
    
    # Shuffle for true randomization across maps (Boston + Singapore)
    random.seed(42)
    random.shuffle(all_tokens)
    
    map_cache = {}
    count = 0
    pbar = tqdm(total=TARGET_SAMPLE_COUNT, desc="🎲 Processing Randomized Samples")
    
    for token in all_tokens:
        if count >= TARGET_SAMPLE_COUNT: break
        
        sample = nusc.get('sample', token)
        loc = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']
        nmap = map_cache.setdefault(loc, NuScenesMap(DATAROOT, loc))
        
        ego = nusc.get('ego_pose', nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token'])
        trans, rot = ego['translation'], ego['rotation']
        
        node_paths = get_reachable_lanes(nmap, trans, rot)
        if not node_paths: continue
        
        inf_map = {t: global_to_local(p, trans, rot) for t, p in node_paths.items()}
        batches = get_batches(nmap, inf_map)
        
        if batches:
            valid_sample_found = False
            for b_idx, (path, ids) in enumerate(batches):
                matrix = calculate_feature_matrix_9col(path, ids, nmap, trans, rot)
                if matrix is not None:
                    torch.save(torch.from_numpy(matrix).float(), os.path.join(PT_DIR, f"{token}_batch{b_idx}.pt"))
                    valid_sample_found = True
            
            if valid_sample_found:
                # Save the images with Batch/Stage titles and 10px green highlighting
                save_batch_visuals_production(nmap, trans, rot, batches, inf_map, token)
                count += 1
                pbar.update(1)
            
            if count % 10 == 0: gc.collect() # Regular memory clearing

if __name__ == "__main__":
    main()
