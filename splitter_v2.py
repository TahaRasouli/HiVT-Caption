import os
import torch
import shutil
import random
from collections import defaultdict
from tqdm import tqdm

# ================= CONFIGURATION =================
SOURCE_DIR = "/mount/studenten/projects/rasoulta/dataset/caption-final"
OUTPUT_DIR = "/mount/studenten/projects/rasoulta/dataset/splits"

# Split Ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# Test will be the remainder

# Removed -1 (Truncated/None) so it doesn't even show up in the logs
LABEL_NAMES = {
    1: "Maintain Lane",
    2: "Turn Left",
    3: "Turn Right",
    4: "Lane Change Left",
    5: "Lane Change Right"
}

def split_dataset_stratified():
    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.pt')]
    
    print(f"[INIT] Reading labels for {len(files)} files to ensure stratified splitting...")
    class_groups = defaultdict(list)
    ignored_count = 0
    
    # 1. Group files by their assigned maneuver class
    for f in tqdm(files, desc="Grouping Classes"):
        file_path = os.path.join(SOURCE_DIR, f)
        try:
            matrix = torch.load(file_path)
            # The label is on the 7th column (index 6)
            label = int(matrix[0, 6].item())
            
            # --- NEW: Drop truncated/invalid samples ---
            if label == -1:
                ignored_count += 1
                continue
                
            class_groups[label].append(f)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    splits = {'train': [], 'val': [], 'test': []}
    random.seed(42)

    # 2. Perform the 80/10/10 split ON EACH CLASS individually
    for label, file_list in class_groups.items():
        random.shuffle(file_list)
        
        train_idx = int(len(file_list) * TRAIN_RATIO)
        val_idx = train_idx + int(len(file_list) * VAL_RATIO)
        
        splits['train'].extend(file_list[:train_idx])
        splits['val'].extend(file_list[train_idx:val_idx])
        splits['test'].extend(file_list[val_idx:])

    # 3. Move the files
    for split_name, file_list in splits.items():
        split_path = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(split_path, exist_ok=True)
        
        for f in tqdm(file_list, desc=f"Moving {split_name} ({len(file_list)} files)"):
            shutil.copy(os.path.join(SOURCE_DIR, f), os.path.join(split_path, f))

    # 4. Print the final stratified distribution
    print(f"\n[DONE] Successfully created stratified splits in {OUTPUT_DIR}")
    print(f"[INFO] Ignored {ignored_count} Truncated/None samples.")
    print("-" * 60)
    print(f"{'Class':<20} | {'Train':<10} | {'Val':<10} | {'Test':<10}")
    print("-" * 60)
    
    for label, name in LABEL_NAMES.items():
        if label not in class_groups: continue
        
        total = len(class_groups[label])
        train_c = int(total * TRAIN_RATIO)
        val_c = int(total * VAL_RATIO)
        test_c = total - train_c - val_c
        
        print(f"{name:<20} | {train_c:<10} | {val_c:<10} | {test_c:<10}")
    print("-" * 60)
    print(f"{'TOTAL':<20} | {len(splits['train']):<10} | {len(splits['val']):<10} | {len(splits['test']):<10}")

if __name__ == "__main__":
    split_dataset_stratified()
