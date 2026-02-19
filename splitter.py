import os
import shutil
import random
from tqdm import tqdm

# ================= CONFIGURATION =================
SOURCE_DIR = "/mount/studenten/projects/rasoulta/dataset/caption-final"
OUTPUT_DIR = "/mount/studenten/projects/rasoulta/dataset/splits"

# Split Ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# Test will be the remainder

def split_dataset():
    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.pt')]
    random.seed(42)
    random.shuffle(files)

    train_idx = int(len(files) * TRAIN_RATIO)
    val_idx = train_idx + int(len(files) * VAL_RATIO)

    splits = {
        'train': files[:train_idx],
        'val': files[train_idx:val_idx],
        'test': files[val_idx:]
    }

    for split_name, file_list in splits.items():
        split_path = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(split_path, exist_ok=True)
        
        for f in tqdm(file_list, desc=f"Moving {split_name}"):
            shutil.copy(os.path.join(SOURCE_DIR, f), os.path.join(split_path, f))

    print(f"\nSuccessfully split {len(files)} files into {OUTPUT_DIR}")
    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

if __name__ == "__main__":
    split_dataset()