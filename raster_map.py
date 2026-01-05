import os
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap

def precompute_raster_maps(nusc_root, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    map_names = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
    resolution = 0.2 # 20cm per pixel
    
    for map_name in map_names:
        print(f"Rasterizing {map_name}...")
        nmap = NuScenesMap(dataroot=nusc_root, map_name=map_name)
        
        # NuScenes maps have a global offset. 
        # canvas_edge is (width, height). 
        # We need the center in global coordinates.
        width, height = nmap.canvas_edge
        center_x, center_y = width / 2, height / 2
        patch_box = (center_x, center_y, height, width)
        
        canvas_size = (int(height / resolution), int(width / resolution))
        layers = ['drivable_area', 'lane', 'lane_connector']
        
        # Generate the mask
        mask = nmap.get_map_mask(patch_box, 0, layers, canvas_size)
        drivable_mask = np.any(mask, axis=0).astype(np.uint8)
        
        # Save mask and meta [width, height, resolution]
        np.save(os.path.join(save_dir, f"{map_name}_mask.npy"), drivable_mask)
        meta = np.array([width, height, resolution])
        np.save(os.path.join(save_dir, f"{map_name}_meta.npy"), meta)
        
        print(f"Saved {map_name} mask. Shape: {drivable_mask.shape}")

if __name__ == "__main__":
    nusc_root = "/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta"
    save_path = os.path.join(nusc_root, "raster_maps")
    precompute_raster_maps(nusc_root, save_path)