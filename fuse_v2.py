import json
import torch
import os
from tqdm import tqdm

# ================= CONFIGURATION =================
JSON_PATH = "project-4.json"  
SOURCE_PT_DIR = "/mount/studenten/projects/rasoulta/dataset/train_processed_labeled"
FINAL_DIR = "/mount/studenten/projects/rasoulta/dataset/caption-final"

os.makedirs(FINAL_DIR, exist_ok=True)

LABEL_MAP = {
    "Maintain Lane": 1.0,
    "Turn Left": 2.0,
    "Turn Right": 3.0,
    "Lane Change Left": 4.0,
    "Lane Change Right": 5.0,
    "None/Other": -1.0  
}

def fuse_labels():
    print(f"[INIT] Reading annotations from {JSON_PATH}...")
    with open(JSON_PATH, 'r') as f:
        annotations = json.load(f)

    # Dictionary to map basename -> label
    # e.g., {"004181f2c1d94571be52c51df1ff5274_r0": 3.0}
    labeled_routes = {}
    
    # 1. Parsing Phase
    for entry in annotations:
        try:
            image_url = entry['data']['image']
            # Safely extract just the filename without the path
            filename = image_url.split('?d=')[-1].split('/')[-1]
            basename = filename.replace(".png", "") # Removes the extension
            
            # SAFEGUARDS for empty tasks
            anns = entry.get('annotations', [])
            if not anns: continue
                
            results = anns[0].get('result', [])
            if not results: continue
                
            maneuver_choice = None
            
            # Find the 'action' label
            for res in results:
                if res.get('from_name') == 'action':
                    choices = res.get('value', {}).get('choices', [])
                    if choices:
                        maneuver_choice = choices[0]
                    break
            
            if maneuver_choice:
                labeled_routes[basename] = LABEL_MAP.get(maneuver_choice, -1.0)
                
        except Exception as e:
            print(f"[WARNING] Failed to parse entry {entry.get('id')}: {e}")
            continue

    print(f"[INFO] Successfully parsed {len(labeled_routes)} labeled routes.")
    success_count = 0
    
    # 2. Processing Phase
    for basename, label_val in tqdm(labeled_routes.items(), desc="Fusing Tensors"):
        source_path = os.path.join(SOURCE_PT_DIR, f"{basename}.pt")
        
        if not os.path.exists(source_path):
            print(f"[WARNING] Missing PT file: {source_path}")
            continue

        # Load the 50x9 tensor
        matrix = torch.load(source_path)
        
        # Apply the label to column index 6 (the 7th column) for all 50 points
        matrix[:, 6] = label_val
        
        # Save to the final directory
        torch.save(matrix, os.path.join(FINAL_DIR, f"{basename}.pt"))
        success_count += 1

    print(f"\n[DONE] Fusion Complete. Saved {success_count} labeled tensors to {FINAL_DIR}")

if __name__ == "__main__":
    fuse_labels()