import os
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap

def precompute_raster_maps(nusc_root, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    map_names = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
    resolution = 0.2 # 0.2m per pixel
    
    for map_name in map_names:
        print(f"Rasterizing {map_name}...")
        nmap = NuScenesMap(dataroot=nusc_root, map_name=map_name)
        
        # 1. Define the patch box to cover the whole map
        # canvas_edge is (width, height). We need (x_center, y_center, height, width)
        width, height = nmap.canvas_edge
        patch_box = (width / 2, height / 2, height, width)
        
        # 2. Define canvas size (pixel dimensions)
        canvas_size = (int(height / resolution), int(width / resolution))
        
        layers = ['drivable_area', 'lane', 'lane_connector']
        
        # 3. Correct call to get_map_mask
        # patch_angle=0 because we want a north-up global mask
        mask = nmap.get_map_mask(patch_box, 0, layers, canvas_size)
        
        # Combine all layers into a single binary "Drivable" mask
        # mask is [num_layers, height, width]
        drivable_mask = np.any(mask, axis=0).astype(np.uint8)
        
        # 4. Save the mask and metadata
        np.save(os.path.join(save_dir, f"{map_name}_mask.npy"), drivable_mask)
        # Meta: [width, height, resolution]
        meta = np.array([width, height, resolution])
        np.save(os.path.join(save_dir, f"{map_name}_meta.npy"), meta)
        
        print(f"Saved {map_name} mask with pixel shape {drivable_mask.shape}")

if __name__ == "__main__":
    nusc_root = "/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta/"
    save_path = os.path.join(nusc_root, "raster_maps")
    precompute_raster_maps(nusc_root, save_path)