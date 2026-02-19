import torch
from torch.utils.data import Dataset
import os

class ManeuverDataset(Dataset):
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.file_list = [f for f in os.listdir(file_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.file_dir, self.file_list[idx]))
        # Features: [x, y, ux, uy, kappa, s] -> Index 0 to 5
        features = data[:, :6].float()
        # Label: Index 6. Mapping 1-5 to 0-4 for Loss Function
        labels = (data[:, 6] - 1).long() 
        return features, labels