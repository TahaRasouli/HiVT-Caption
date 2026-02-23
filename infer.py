import os
import json
import torch
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from nuscenes.map_expansion.map_api import NuScenesMap
from models.cvae import CVAE

# =========================================================
# CONFIGURATION
# =========================================================
NUSCENES_DATAROOT = "/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta/"
DATA_DIR = "/mount/studenten/projects/rasoulta/dataset/train_processed"
CHECKPOINT = "/mount/studenten/projects/rasoulta/checkpoints/vae_best/checkpoints/epoch=47-step=15888.ckpt"
NUM_SAMPLES = 5
K = 6
DT = 0.1
ACTOR_FILTER_DIST = 15.0  # Filter actors > 15m away
SIGNAL_FILTER_DIST = 10.0 # Filter signals > 10m away

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
map_cache = {}

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
    # Unwrap to handle -180 to 180 transitions
    diff = np.degrees(np.unwrap([yaw_start, yaw_end])[1] - yaw_start)
    return diff

# =========================================================
# SCENE ANALYZER
# =========================================================

class SceneAnalyzer:
    def __init__(self, nmap):
        self.nmap = nmap

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

    def get_structured_lanes(self, ego_pos):
        ego_lane = self.nmap.get_closest_lane(ego_pos[0], ego_pos[1], radius=2.0)
        if not ego_lane: return None, []
        
        block_token = None
        for rb in self.nmap.road_block:
            if ego_lane in rb.get('lane_tokens', []):
                block_token = rb['token']
                break
        
        if not block_token: return ego_lane, [ego_lane]
        lanes = self.nmap.get('road_block', block_token).get('lane_tokens', [])
        
        try:
            ego_rec = self.nmap.get('lane', ego_lane)
            n1 = self.nmap.get('node', ego_rec['exterior_node_tokens'][0])
            n2 = self.nmap.get('node', ego_rec['exterior_node_tokens'][-1])
            dx, dy = n2['x'] - n1['x'], n2['y'] - n1['y']
            perp_x, perp_y = -dy, dx 

            lane_scores = []
            for l in lanes:
                rec = self.nmap.get('lane', l)
                ns = self.nmap.get('node', rec['exterior_node_tokens'][0])
                ne = self.nmap.get('node', rec['exterior_node_tokens'][-1])
                mid_x, mid_y = (ns['x']+ne['x'])/2, (ns['y']+ne['y'])/2
                score = mid_x*perp_x + mid_y*perp_y
                lane_scores.append((score, l))
            
            lane_scores.sort(key=lambda x: x[0])
            sorted_lanes = [x[1] for x in lane_scores]
            return ego_lane, sorted_lanes
        except: return ego_lane, lanes

    def group_actors_by_lane(self, data, origin, theta, ego_lane, sorted_lanes):
        groups = {}
        for i, l in enumerate(sorted_lanes):
            label = f"Lane {i+1}"
            if l == ego_lane: label += " (Ego Lane)"
            groups[l] = {"label": label, "lane_id": l, "actors": []}
        
        other_actors = []
        current_pos = data.positions[:, 19].cpu().numpy()
        ego_idx = int(data.av_index) if hasattr(data, 'av_index') else 0

        for i in range(current_pos.shape[0]):
            if i == ego_idx: continue
            
            # Distance Filter
            dist = float(np.linalg.norm(current_pos[i]))
            
            # --- FILTER: ONLY ACTORS WITHIN 15M ---
            if dist > ACTOR_FILTER_DIST: continue

            pos_global = local_to_global(current_pos[i], origin, theta).flatten()
            act_lane = self.nmap.get_closest_lane(pos_global[0], pos_global[1], radius=1.0)
            
            actor_data = {"id": int(i), "dist": round(dist, 1)}

            if act_lane in groups:
                groups[act_lane]["actors"].append(actor_data)
            else:
                other_actors.append(actor_data)

        output = []
        for l in sorted_lanes:
            if groups[l]["actors"] or l == ego_lane:
                output.append(groups[l])
        
        if other_actors:
            output.append({
                "label": "Non-Lane / Other",
                "lane_id": "n/a",
                "actors": other_actors
            })
        return output

    def find_signals(self, pos, dist=40):
        patch = [pos[0]-dist, pos[1]-dist, pos[0]+dist, pos[1]+dist]
        try:
            recs = self.nmap.get_records_in_patch(patch, ['stop_line', 'traffic_light'], mode='intersect')
            found = []
            for t in recs.get('stop_line', []):
                sl = self.nmap.get('stop_line', t)
                poly = self.nmap.extract_polygon(sl['polygon_token'])
                pts = None
                if isinstance(poly, ShapelyPolygon) and not poly.is_empty:
                    pts = np.array(poly.exterior.coords)
                elif isinstance(poly, (list, np.ndarray)):
                    pts = np.array(poly)
                if pts is not None and len(pts) > 0:
                    d = float(np.linalg.norm(pos - np.mean(pts[:,:2], axis=0)))
                    
                    # --- STRICT DISTANCE CHECK ---
                    if d < dist:
                        found.append({"type": sl['stop_line_type'], "dist": round(d, 1)})
            return sorted(found, key=lambda x: x['dist'])
        except: return []

# =========================================================
# MAIN
# =========================================================

def main():
    print(f"Loading Model...")
    model = CVAE.load_from_checkpoint(CHECKPOINT, map_location=device).to(device).eval()
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.pt')])[:NUM_SAMPLES]

    for fname in files:
        try:
            data = torch.load(os.path.join(DATA_DIR, fname), map_location=device)
            nmap = get_map(data.city)
            if not nmap: continue
            
            analyzer = SceneAnalyzer(nmap)
            origin = data.origin[0].cpu().numpy()
            theta = data.theta.cpu().item()

            ego_lane, sorted_lanes = analyzer.get_structured_lanes(origin)
            intersection = analyzer.classify_intersection(ego_lane)
            actor_groups = analyzer.group_actors_by_lane(data, origin, theta, ego_lane, sorted_lanes)
            
            # --- SIGNAL LOOKUP (10 METERS) ---
            signals = analyzer.find_signals(origin, dist=SIGNAL_FILTER_DIST)

            # Generate Trajectories
            context = model(data)
            context = context.reshape(-1, model.hparams.embed_dim)
            ego_context = context[int(data.av_index)].repeat(K, 1)
            traj_local, _ = model.decoder(ego_context, None)
            traj_local = traj_local.reshape(K, 30, 2).detach().cpu().numpy()

            candidates = []
            for k in range(K):
                # Full Global Trajectory
                t_glob = local_to_global(traj_local[k], origin, theta)
                
                # --- LATERAL STATUS (Overall) ---
                start_lane = ego_lane
                end_lane = analyzer.nmap.get_closest_lane(t_glob[-1,0], t_glob[-1,1], 2.0)
                
                lateral_status = "unknown"
                if end_lane == start_lane:
                    lateral_status = "keep_lane"
                elif end_lane in sorted_lanes:
                    try:
                        curr_idx = sorted_lanes.index(start_lane)
                        new_idx = sorted_lanes.index(end_lane)
                        if new_idx < curr_idx: lateral_status = "lane_change_right"
                        else: lateral_status = "lane_change_left"
                    except: lateral_status = "lane_change_general"
                else:
                    lateral_status = "turning/exit"

                # --- PER SECOND ANALYSIS ---
                progression = []
                # Timesteps: 0, 10, 20, 30
                
                for i in range(3):
                    # Define segment indices (0-10, 10-20, 20-30)
                    idx_start = i * 10
                    idx_end = (i + 1) * 10 - 1 # Use 9, 19, 29 as endpoints
                    
                    if idx_end >= 30: idx_end = 29

                    # Calculate Vectors
                    # Start Vector (Instantaneous at start of second)
                    if idx_start + 1 < 30:
                        v_start = t_glob[idx_start+1] - t_glob[idx_start]
                    else:
                        v_start = t_glob[idx_start] - t_glob[idx_start-1]

                    # End Vector (Instantaneous at end of second)
                    if idx_end + 1 < 30:
                        v_end = t_glob[idx_end+1] - t_glob[idx_end]
                    else:
                        v_end = t_glob[idx_end] - t_glob[idx_end-1]
                    
                    yaw1 = get_yaw(v_start)
                    yaw2 = get_yaw(v_end)
                    
                    yaw_delta = float(get_yaw_change(yaw1, yaw2))
                    
                    if abs(yaw_delta) < 5.0: m_type = "maintain_heading"
                    elif yaw_delta > 5.0: m_type = "turn_left"
                    else: m_type = "turn_right"
                    
                    progression.append({
                        "second": i+1,
                        "yaw_change_deg": round(yaw_delta, 1),
                        "maneuver": m_type
                    })

                speed_val = float(np.mean(np.linalg.norm(np.diff(traj_local[k], axis=0), axis=1)/DT))

                candidates.append({
                    "id": k, 
                    "lateral_status": lateral_status,
                    "avg_speed": round(speed_val, 2),
                    "progression": progression
                })

            output = {
                "file": fname,
                "ego_state": {
                    "lane_index": sorted_lanes.index(ego_lane) + 1 if ego_lane in sorted_lanes else -1,
                    "total_lanes": len(sorted_lanes),
                    "intersection_ahead": intersection
                },
                "surrounding_actors_within_15m": actor_groups,
                "signals_within_10m": signals,
                "predictions": candidates
            }
            print(json.dumps(output, indent=2))

        except Exception as e:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()