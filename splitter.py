import os
import shutil
import random
import torch
from tqdm import tqdm
from collections import defaultdict

# ================= CONFIGURATION =================
SOURCE_DIR = "/mount/studenten/projects/rasoulta/dataset/caption-final"
OUTPUT_DIR = "/mount/studenten/projects/rasoulta/dataset/splits"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# Test will be the remainder

def get_major_class(file_path):
    """Loads the file and finds the most frequent label (excluding padding)."""
    data = torch.load(file_path)
    labels = data[:, 6].long()
    # Remove any -100 padding if it exists at this stage
    valid_labels = labels[labels != -100]
    if len(valid_labels) == 0:
        return 0
    # Returns the most common maneuver in this specific file
    return torch.mode(valid_labels).values.item()

def split_dataset():
    # 1. Categorize all files by their majority class
    print("🔍 Scanning dataset classes for stratification...")
    class_map = defaultdict(list)
    all_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.pt')]
    
    for f in tqdm(all_files):
        major_class = get_major_class(os.path.join(SOURCE_DIR, f))
        class_map[major_class].append(f)

    splits = {'train': [], 'val': [], 'test': []}
    random.seed(42)

    # 2. Split each class individually to maintain ratios
    for class_id, files in class_map.items():
        random.shuffle(files)
        
        n = len(files)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)
        
        splits['train'].extend(files[:train_end])
        splits['val'].extend(files[train_end:val_end])
        splits['test'].extend(files[val_end:])
        
        print(f"Class {class_id}: {n} files -> {train_end} Train, {val_end-train_end} Val, {n-val_end} Test")

    # 3. Clear and create output directories
    if os.path.exists(OUTPUT_DIR):
        print(f"⚠️ Warning: Cleaning old splits in {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # 4. Copy files
    for split_name, file_list in splits.items():
        dest_path = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(dest_path, exist_ok=True)
        
        for f in tqdm(file_list, desc=f"📦 Copying {split_name}"):
            shutil.copy(os.path.join(SOURCE_DIR, f), os.path.join(dest_path, f))

    print(f"\n✅ Stratified split complete!")
    print(f"Total Files: {len(all_files)}")
    print(f"Final Counts -> Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

if __name__ == "__main__":
    split_dataset()