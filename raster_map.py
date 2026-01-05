import os
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.bitmap import BitMap

def precompute_raster_maps(nusc_root, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    map_names = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
    
    for map_name in map_names:
        print(f"Rasterizing {map_name}...")
        nmap = NuScenesMap(dataroot=nusc_root, map_name=map_name)
        
        # We use a 0.2m per pixel resolution for high accuracy
        # This covers the entire map area as a binary mask
        canvas_size = (int(nmap.canvas_edge[0] * 5), int(nmap.canvas_edge[1] * 5))
        
        # Layer names that count as 'On-Road'
        layers = ['drivable_area', 'lane', 'lane_connector']
        
        # get_map_mask returns [layers, height, width]
        mask = nmap.get_map_mask(nmap.canvas_edge, 0, layers, canvas_size)
        
        # Combine all layers into a single binary "Drivable" mask
        drivable_mask = np.any(mask, axis=0).astype(np.uint8)
        
        # Save the mask and the metadata (so we can map global coords back to pixels)
        np.save(os.path.join(save_dir, f"{map_name}_mask.npy"), drivable_mask)
        print(f"Saved {map_name} mask with shape {drivable_mask.shape}")

# Usage
nusc_root = "/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta/v1.0-trainval/"
save_path = os.path.join(nusc_root, "raster_maps")
precompute_raster_maps(nusc_root, save_path)
