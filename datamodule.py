import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataset import ManeuverDataset

# def maneuver_collate_fn(batch):
#     """
#     Pads sequences of different lengths (e.g. 50, 100, 150 points) 
#     so they can be processed in a single batch.
#     """
#     features = [item[0] for item in batch]
#     labels = [item[1] for item in batch]

#     # Pad features with 0.0
#     features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    
#     # Pad labels with -100 so CrossEntropyLoss ignores them
#     labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

#     return features_padded, labels_padded


def maneuver_collate_fn(batch):
    # batch is a list of tuples: [(feats, labels, filename), ...]
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Check if your dataset __getitem__ returns 3 items
    # If it does, we grab the filenames here:
    filenames = [item[2] for item in batch] if len(batch[0]) > 2 else None

    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    # Return 3 items to match the model's expected unpacking
    return features_padded, labels_padded, filenames

class ManeuverDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size=16, num_workers=8):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Augment only the training set
        self.train_ds = ManeuverDataset(f"{self.data_root}/train", augment=True)
        self.val_ds = ManeuverDataset(f"{self.data_root}/val", augment=False)
        self.test_ds = ManeuverDataset(f"{self.data_root}/test", augment=False)
        self.test_ds = ManeuverDataset(f"{self.data_root}/test")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=maneuver_collate_fn,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            collate_fn=maneuver_collate_fn,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            collate_fn=maneuver_collate_fn,
            num_workers=self.num_workers
        )