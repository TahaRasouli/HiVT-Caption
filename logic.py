import os
import json
import torch
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from models.cvae import CVAE

# =========================================================
# CONFIGURATION
# =========================================================
NUSCENES_DATAROOT = "/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta/"
DATA_DIR = "/mount/studenten/projects/rasoulta/dataset/train_processed"
CHECKPOINT = "/mount/studenten/projects/rasoulta/checkpoints/vae_best/checkpoints/epoch=47-step=15888.ckpt"
NUSCENES_VERSION = "v1.0-trainval" 

TARGET_SAMPLES = 10     
K = 6                     
DT = 0.1                  
ACTOR_FILTER_DIST = 15.0  
SIGNAL_FILTER_DIST = 40.0 
STATIONARY_THRESHOLD = 0.5 # Displacement (meters) per second to be stationary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global Caches
map_cache = {}
nusc_cache = None

def get_nusc():
    global nusc_cache
    if nusc_cache is None:
        print(f"Loading NuScenes {NUSCENES_VERSION} DB (this takes a moment)...")
        try:
            nusc_cache = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)
        except Exception as e:
            print(f"Failed to load NuScenes DB: {e}")
            return None
    return nusc_cache

def get_map(location):
    if location not in map_cache:
        try:
            map_cache[location] = NuScenesMap(dataroot=NUSCENES_DATAROOT, map_name=location)
        except Exception:
            return None
    return map_cache[location]

# =========================================================
# UTILITIES
# =========================================================

def local_to_global(coords_local, origin, theta):
    if isinstance(coords_local, torch.Tensor): coords_local = coords_local.cpu().numpy()
    if isinstance(origin, torch.Tensor): origin = origin.cpu().numpy()
    if isinstance(theta, torch.Tensor): theta = theta.cpu().item()

    cos, sin = np.cos(theta), np.sin(theta)
    rot_inv = np.array([[cos, -sin], [sin, cos]]) 
    if coords_local.ndim == 1: coords_local = coords_local.reshape(1, 2)
    return (coords_local @ rot_inv.T) + origin

def get_yaw(vector):
    return np.arctan2(vector[1], vector[0])

def get_yaw_change(yaw_start, yaw_end):
    diff = np.degrees(np.unwrap([yaw_start, yaw_end])[1] - yaw_start)
    return diff

# =========================================================
# SCENE ANALYZER
# =========================================================

class SceneAnalyzer:
    def __init__(self, nmap, nusc):
        self.nmap = nmap
        self.nusc = nusc

    def get_heading(self, lane_token):
        try:
            rec = self.nmap.get('lane', lane_token)
            n1 = self.nmap.get('node', rec['exterior_node_tokens'][0])
            n2 = self.nmap.get('node', rec['exterior_node_tokens'][-1])
            return np.arctan2(n2['y'] - n1['y'], n2['x'] - n1['x'])
        except: return 0.0

    def get_structured_lanes(self, ego_pos):
        ego_lane = self.nmap.get_closest_lane(ego_pos[0], ego_pos[1], radius=2.0)
        if not ego_lane: return None, [], None
        
        block_token = None
        for rb in self.nmap.road_block:
            if ego_lane in rb.get('lane_tokens', []):
                block_token = rb['token']
                break
        
        ego_heading = self.get_heading(ego_lane)
        x, y = ego_pos[0], ego_pos[1]
        patch = [x-8, y-8, x+8, y+8]
        try:
            candidates = self.nmap.get_records_in_patch(patch, ['lane'], mode='intersect')['lane']
        except:
            return ego_lane, [ego_lane], block_token

        parallel_lanes = []
        for l in candidates:
            h = self.get_heading(l)
            diff = abs(np.arctan2(np.sin(h-ego_heading), np.cos(h-ego_heading)))
            if diff < np.radians(30): parallel_lanes.append(l)
        
        parallel_lanes = list(set(parallel_lanes))
        if not parallel_lanes: return ego_lane, [ego_lane], block_token

        perp_x, perp_y = -np.sin(ego_heading), np.cos(ego_heading)
        lane_scores = []
        for l in parallel_lanes:
            rec = self.nmap.get('lane', l)
            n1 = self.nmap.get('node', rec['exterior_node_tokens'][0])
            n2 = self.nmap.get('node', rec['exterior_node_tokens'][-1])
            mid_x, mid_y = (n1['x']+n2['x'])/2, (n1['y']+n2['y'])/2
            score = mid_x*perp_x + mid_y*perp_y
            lane_scores.append((score, l))
        
        lane_scores.sort(key=lambda x: x[0])
        sorted_lanes = [x[1] for x in lane_scores]
        return ego_lane, sorted_lanes, block_token

    def classify_intersection(self, lane_id):
        try:
            outgoing = self.nmap.get_outgoing_lane_ids(lane_id)
            if not outgoing: return {"is_ahead": False, "type": "None"}
            next_lane = outgoing[0]
            path = self.nmap.get_arcline_path(next_lane)
            pt = path[0]['start_pose'][:2]
            layers = self.nmap.layers_on_point(pt[0], pt[1])
            if layers.get('road_segment'):
                seg = self.nmap.get('road_segment', layers['road_segment'])
                if seg['is_intersection']:
                    return {"is_ahead": True, "type": self._count_arms(seg['polygon_token'])}
        except: pass
        return {"is_ahead": False, "type": "None"}

    def _count_arms(self, poly_token):
        try:
            poly = self.nmap.extract_polygon(poly_token)
            if not hasattr(poly, 'bounds'): return "Unknown"
            minx, miny, maxx, maxy = poly.bounds
            patch = [minx-2, miny-2, maxx+2, maxy+2]
            blocks = self.nmap.get_records_in_patch(patch, ['road_block'], mode='intersect')
            count = len(set(blocks.get('road_block', [])))
            if count <= 4: return "Fork/Merge"
            if 5 <= count <= 6: return "T-Junction"
            if 7 <= count <= 8: return "Square Intersection"
            return "Complex Junction"
        except: return "Unknown"

    def match_actor_type(self, pos_global, sample_token):
        try:
            sample = self.nusc.get('sample', sample_token)
            min_dist = 2.0 
            best_cat = "unknown"
            for ann_token in sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                tx, ty = ann['translation'][0], ann['translation'][1]
                dist = np.hypot(pos_global[0]-tx, pos_global[1]-ty)
                if dist < min_dist:
                    min_dist = dist
                    best_cat = ann['category_name'] 
            if 'human' in best_cat: return "pedestrian"
            if 'vehicle' in best_cat:
                if 'truck' in best_cat: return "truck"
                if 'bicycle' in best_cat: return "cyclist"
                if 'bus' in best_cat: return "bus"
                return "vehicle"
            return "static_object"
        except: return "unknown"

    def group_actors_by_lane(self, data, origin, theta, ego_lane, sorted_lanes, sample_token):
        groups = {l: {"label": f"Lane {i+1}{' (Ego Lane)' if l == ego_lane else ''}", "lane_id": l, "actors": []} for i, l in enumerate(sorted_lanes)}
        other_actors = []
        current_pos = data.positions[:, 19].cpu().numpy()
        ego_idx = int(data.av_index) if hasattr(data, 'av_index') else 0

        for i in range(current_pos.shape[0]):
            if i == ego_idx: continue
            dist = float(np.linalg.norm(current_pos[i]))
            if dist > ACTOR_FILTER_DIST: continue
            pos_global = local_to_global(current_pos[i], origin, theta).flatten()
            actor_type = self.match_actor_type(pos_global, sample_token)
            act_lane = self.nmap.get_closest_lane(pos_global[0], pos_global[1], radius=1.0)
            actor_data = {"id": int(i), "type": actor_type, "dist": round(dist, 1)}
            if act_lane in groups: groups[act_lane]["actors"].append(actor_data)
            else: other_actors.append(actor_data)

        output = [groups[l] for l in sorted_lanes if groups[l]["actors"] or l == ego_lane]
        if other_actors: output.append({"label": "Non-Lane / Other", "lane_id": "n/a", "actors": other_actors})
        return output

    def find_signals(self, pos, ego_road_block, dist=40):
        patch = [pos[0]-dist, pos[1]-dist, pos[0]+dist, pos[1]+dist]
        try:
            recs = self.nmap.get_records_in_patch(patch, ['stop_line', 'traffic_light'], mode='intersect')
            found = []
            for t in recs.get('traffic_light', []):
                tl = self.nmap.get('traffic_light', t)
                is_relevant = (tl['from_road_block_token'] == ego_road_block)
                colors = "/".join(set([item['color'] for item in tl['items']]))
                x, y = tl['pose']['tx'], tl['pose']['ty']
                d = float(np.hypot(pos[0]-x, pos[1]-y))
                if d < dist:
                    found.append({"object": "traffic_light", "on_ego_lane": is_relevant, "color": colors, "dist": round(d, 1)})
            for t in recs.get('stop_line', []):
                sl = self.nmap.get('stop_line', t)
                poly = self.nmap.extract_polygon(sl['polygon_token'])
                pts = np.array(poly.exterior.coords) if hasattr(poly, 'exterior') else np.array(poly)
                d = float(np.linalg.norm(pos - np.mean(pts[:,:2], axis=0)))
                if d < 10.0:
                    found.append({"object": "stop_line", "type": sl['stop_line_type'], "dist": round(d, 1)})
            return sorted(found, key=lambda x: x['dist'])
        except: return []

# =========================================================
# MAIN
# =========================================================

def main():
    print(f"Loading Model...")
    model = CVAE.load_from_checkpoint(CHECKPOINT, map_location=device).to(device).eval()
    nusc = get_nusc()
    if nusc is None: return

    all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.pt')])
    processed_count = 0
    print(f"Scanning for {TARGET_SAMPLES} samples with >= 3 drivable lanes and diverse intents...")

    for fname in all_files:
        if processed_count >= TARGET_SAMPLES: break
        try:
            data = torch.load(os.path.join(DATA_DIR, fname), map_location=device)
            nmap = get_map(data.city)
            if not nmap: continue
            
            analyzer = SceneAnalyzer(nmap, nusc)
            origin = data.origin[0].cpu().numpy()
            theta = data.theta.cpu().item()
            sample_token = fname.replace('.pt', '')

            ego_lane, sorted_lanes, ego_block = analyzer.get_structured_lanes(origin)
            if len(sorted_lanes) < 3: continue

            # Inference
            with torch.no_grad():
                context = model(data).reshape(-1, model.hparams.embed_dim)
                ego_context = context[int(data.av_index)].repeat(K, 1)
                traj_local, _ = model.decoder(ego_context, None)
                traj_local = traj_local.reshape(K, 30, 2).detach().cpu().numpy()

            candidates = []
            distinct_statuses = set()

            for k in range(K):
                t_glob = local_to_global(traj_local[k], origin, theta)
                t_loc = traj_local[k]
                end_lane = analyzer.nmap.get_closest_lane(t_glob[-1,0], t_glob[-1,1], 2.0)
                
                # --- LATERAL STATUS LOGIC ---
                lateral_status = "unknown"
                if end_lane == ego_lane: lateral_status = "keep_lane"
                elif end_lane in sorted_lanes:
                    lateral_status = "lane_change_right" if sorted_lanes.index(end_lane) < sorted_lanes.index(ego_lane) else "lane_change_left"
                else:
                    outgoing = analyzer.nmap.get_outgoing_lane_ids(ego_lane)
                    is_succ = any(end_lane == out or end_lane in analyzer.nmap.get_outgoing_lane_ids(out) for out in outgoing)
                    if is_succ: lateral_status = "keep_lane_crossing_intersection"
                    else:
                        road_heading = analyzer.get_heading(ego_lane)
                        vec_traj = t_glob[-1] - t_glob[-5]
                        angle_diff = float(get_yaw_change(road_heading, np.arctan2(vec_traj[1], vec_traj[0])))
                        if end_lane is None: lateral_status = "off_road_trajectory"
                        elif abs(angle_diff) > 45.0: lateral_status = "turn_left_at_intersection" if angle_diff > 0 else "turn_right_at_intersection"
                        else: lateral_status = "turning/exit_general"

                distinct_statuses.add(lateral_status)

                # --- MANEUVER LOGIC (INCLUDING STATIONARY) ---
                progression = []
                for i in range(3):
                    idx_s, idx_e = i * 10, (i + 1) * 10 - 1
                    dist_covered = np.linalg.norm(t_loc[idx_e] - t_loc[idx_s])
                    
                    if dist_covered < STATIONARY_THRESHOLD:
                        m_type = "stationary"
                        yaw_delta = 0.0
                    else:
                        v_s = t_glob[idx_s+1]-t_glob[idx_s] if idx_s+1<30 else t_glob[idx_s]-t_glob[idx_s-1]
                        v_e = t_glob[idx_e]-t_glob[idx_e-1]
                        yaw_delta = float(get_yaw_change(get_yaw(v_s), get_yaw(v_e)))
                        m_type = "maintain_heading" if abs(yaw_delta) < 5.0 else ("turn_left" if yaw_delta > 5.0 else "turn_right")
                    
                    progression.append({"sec": i+1, "dist_m": round(float(dist_covered), 2), "yaw_d": round(yaw_delta, 1), "maneuver": m_type})

                candidates.append({"id": k, "lateral_status": lateral_status, "avg_speed": round(float(np.mean(np.linalg.norm(np.diff(t_loc, axis=0), axis=1)/DT)), 2), "progression": progression})

            # Diversity Filter
            if len(distinct_statuses) < 2: continue
            
            processed_count += 1
            output = {
                "file": fname,
                "ego_state": {"lane_index": sorted_lanes.index(ego_lane) + 1, "total_lanes": len(sorted_lanes), "intersection_ahead": analyzer.classify_intersection(ego_lane)},
                "surrounding_actors_within_15m": analyzer.group_actors_by_lane(data, origin, theta, ego_lane, sorted_lanes, sample_token),
                "signals_within_10m": analyzer.find_signals(origin, ego_block),
                "predictions": candidates
            }
            print(json.dumps(output, indent=2))

        except Exception: pass

if __name__ == "__main__":
    main()