import json
import torch
import os
from tqdm import tqdm

# ================= CONFIGURATION =================
JSON_PATH = "export backup.json"  # The file you downloaded from Label Studio
SOURCE_PT_DIR = "/mount/studenten/projects/rasoulta/dataset/train_processed_labeled"
FINAL_DIR = "/mount/studenten/projects/rasoulta/dataset/caption-final"

os.makedirs(FINAL_DIR, exist_ok=True)

# Define the numerical mapping for your classes
# Note: 0 is usually reserved for "padding" or "unknown" in many ML models
# Define the numerical mapping
LABEL_MAP = {
    "Maintain Lane": 1.0,
    "Turn Left": 2.0,
    "Turn Right": 3.0,
    "Lane Change Left": 4.0,
    "Lane Change Right": 5.0,
    "None/Other": -1.0  # Temporary flag for truncation
}

def fuse_and_truncate():
    print(f"[INIT] Reading annotations from {JSON_PATH}...")
    with open(JSON_PATH, 'r') as f:
        annotations = json.load(f)

    # Organize annotations by batch to handle sequential truncation
    # Key: sampleToken_batchX -> {stageY: label_id}
    batch_data = {}
    
    for entry in annotations:
        try:
            image_url = entry['data']['image']
            filename = image_url.split('/')[-1].split('?d=')[-1]
            parts = filename.replace(".png", "").split("_")
            
            token_batch = f"{parts[0]}_{parts[1]}" # e.g., token_batch0
            stage_idx = int(parts[2].replace("stage", "")) # e.g., 1
            
            results = entry['annotations'][0]['result']
            maneuver_choice = None
            for res in results:
                if res.get('from_name') == 'maneuver':
                    maneuver_choice = res['value']['choices'][0]
                    break
            
            if maneuver_choice:
                if token_batch not in batch_data:
                    batch_data[token_batch] = {}
                batch_data[token_batch][stage_idx] = LABEL_MAP[maneuver_choice]
        except:
            continue

    success_count = 0
    
    # Process organized batches
    for base_name, stages in tqdm(batch_data.items(), desc="Truncating & Saving"):
        source_path = os.path.join(SOURCE_PT_DIR, f"{base_name}.pt")
        if not os.path.exists(source_path):
            continue

        # Logic: Find where the "None" (-1.0) occurs
        # If stage 1 is None -> discard whole batch
        # If stage 2 is None -> keep stage 1, cut rest
        # If stage 3 is None -> keep stage 1 & 2, cut rest
        
        sorted_stages = sorted(stages.keys())
        valid_stages_count = 0
        
        for s in sorted_stages:
            if stages[s] == -1.0:
                break # Stop adding stages once "None" is found
            valid_stages_count += 1
            
        if valid_stages_count == 0:
            continue # Whole batch was "None" or started with "None"

        # Load the full tensor [Points, Columns]
        # In your case, usually 50 points per stage or 50 points total?
        # Assuming the matrix was saved as the full stitched trajectory (e.g., 150 points for 3 stages)
        matrix = torch.load(source_path)
        points_per_stage = matrix.shape[0] // len(sorted_stages)
        
        # Calculate truncation point
        cutoff_idx = valid_stages_count * points_per_stage
        final_matrix = matrix[:cutoff_idx].clone()
        
        # Apply labels to the points within each stage segment
        for i in range(valid_stages_count):
            start = i * points_per_stage
            end = (i + 1) * points_per_stage
            stage_key = sorted_stages[i]
            final_matrix[start:end, 6] = stages[stage_key]

        torch.save(final_matrix, os.path.join(FINAL_DIR, f"{base_name}.pt"))
        success_count += 1

    print(f"Fusion & Truncation Complete. Saved {success_count} batches to {FINAL_DIR}")

if __name__ == "__main__":
    fuse_and_truncate()