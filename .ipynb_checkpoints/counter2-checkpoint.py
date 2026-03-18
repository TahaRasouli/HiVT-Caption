import os
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm

# ================= CONFIGURATION =================
FINAL_DIR = "/mount/studenten/projects/rasoulta/dataset/caption-final"

# Matching the mapping used in your Fusion script
LABEL_MAP_INV = {
    1.0: "Maintain Lane",
    2.0: "Turn Left",
    3.0: "Turn Right",
    4.0: "Lane Change Left",
    5.0: "Lane Change Right",
    0.0: "None/Other",
    -1.0: "Truncated/None"
}

def analyze_final_dataset(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        return

    file_list = [f for f in os.listdir(directory) if f.endswith('.pt')]
    
    # We'll count occurrences of each class
    # Note: Since one .pt file can contain multiple stages/classes, 
    # we count unique classes per file and total points per class.
    class_counts = Counter()
    total_points_per_class = Counter()

    print(f"Analyzing {len(file_list)} files in {directory}...")

    for fname in tqdm(file_list, desc="Reading Tensors"):
        path = os.path.join(directory, fname)
        try:
            # Load tensor [Points, Columns]
            matrix = torch.load(path)
            
            # The label is in index 6 (7th column)
            labels = matrix[:, 6].numpy()
            
            # Find unique labels in this specific file/batch
            unique_in_file = np.unique(labels)
            for lbl in unique_in_file:
                class_name = LABEL_MAP_INV.get(float(lbl), f"Unknown({lbl})")
                class_counts[class_name] += 1
            
            # Count total points for each class (useful for loss weighting)
            for lbl in labels:
                class_name = LABEL_MAP_INV.get(float(lbl), f"Unknown({lbl})")
                total_points_per_class[class_name] += 1
                
        except Exception as e:
            print(f"Could not read {fname}: {e}")

    # ================= RESULTS =================
    print("\n" + "="*50)
    print(f"{'Maneuver Class':<20} | {'Files':<10} | {'Total Points'}")
    print("-" * 50)
    
    # Sort by class ID or name for readability
    for label_val in sorted(LABEL_MAP_INV.keys()):
        name = LABEL_MAP_INV[label_val]
        if name in class_counts or name in total_points_per_class:
            print(f"{name:<20} | {class_counts[name]:<10} | {total_points_per_class[name]}")
            
    print("="*50)
    print(f"Total valid .pt files: {len(file_list)}")

if __name__ == "__main__":
    analyze_final_dataset(FINAL_DIR)