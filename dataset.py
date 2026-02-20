import torch
from torch.utils.data import Dataset
import os

class ManeuverDataset(Dataset):
    def __init__(self, file_dir, augment=False):
        self.file_dir = file_dir
        self.file_list = sorted([f for f in os.listdir(file_dir) if f.endswith('.pt')])
        self.augment = augment

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.file_dir, self.file_list[idx]))
        
        # 1. Get original 6 features: [s, d, v, a, theta, kappa]
        feats = data[:, :6].float()
        labels = (data[:, 6] - 1).long()

        # 2. FEATURE ENGINEERING: Calculate deltas for (v, theta, kappa)
        # This expands features from 6 to 9
        deltas = torch.zeros_like(feats[:, :3]) 
        # diff: current_step - previous_step
        deltas[1:] = feats[1:, [2, 4, 5]] - feats[:-1, [2, 4, 5]]
        
        # Size becomes [Seq, 9]
        combined_feats = torch.cat([feats, deltas], dim=-1)

        # 3. Apply Mirroring Augmentation if needed
        if self.augment and torch.rand(1) < 0.5:
            # Flip lateral geometry (d, theta, kappa, d_theta, d_kappa)
            combined_feats[:, 1] *= -1 # d
            combined_feats[:, 4] *= -1 # theta
            combined_feats[:, 5] *= -1 # kappa
            combined_feats[:, 7] *= -1 # d_theta
            combined_feats[:, 8] *= -1 # d_kappa
            
            # Swap labels logic
            new_labels = labels.clone()
            new_labels[labels == 1] = 2
            new_labels[labels == 2] = 1
            new_labels[labels == 3] = 4
            new_labels[labels == 4] = 3
            labels = new_labels

        return combined_feats, labels, self.file_list[idx]

    def __len__(self):
        return len(self.file_list)