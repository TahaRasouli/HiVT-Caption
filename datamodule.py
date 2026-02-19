import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataset import ManeuverDataset, StatefulManeuverDataset

def maneuver_collate_fn(batch):
    """
    Pads sequences of different lengths (e.g. 50, 100, 150 points) 
    so they can be processed in a single batch.
    """
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Pad features with 0.0
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    
    # Pad labels with -100 so CrossEntropyLoss ignores them
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return features_padded, labels_padded

def stateful_collate_fn(batch):
    # batch = [(features, labels, scene_id), ...]
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    scene_ids = [item[2] for item in batch]

    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return features_padded, labels_padded, scene_ids

# class ManeuverDataModule(pl.LightningDataModule):
#     # FIXED: Added num_workers to __init__
#     def __init__(self, data_root, batch_size=16, num_workers=8):
#         super().__init__()
#         self.data_root = data_root
#         self.batch_size = batch_size
#         self.num_workers = num_workers

#     def setup(self, stage=None):
#         self.train_ds = ManeuverDataset(f"{self.data_root}/train")
#         self.val_ds = ManeuverDataset(f"{self.data_root}/val")
#         self.test_ds = ManeuverDataset(f"{self.data_root}/test")

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_ds, 
#             batch_size=self.batch_size, 
#             shuffle=True, 
#             collate_fn=maneuver_collate_fn,
#             num_workers=self.num_workers
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_ds, 
#             batch_size=self.batch_size, 
#             collate_fn=maneuver_collate_fn,
#             num_workers=self.num_workers
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.test_ds, 
#             batch_size=self.batch_size, 
#             collate_fn=maneuver_collate_fn,
#             num_workers=self.num_workers
#         )

class ManeuverDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size=1, num_workers=8):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Using the new Stateful Dataset class
        self.train_ds = StatefulManeuverDataset(f"{self.data_root}/train")
        self.val_ds = StatefulManeuverDataset(f"{self.data_root}/val")
        self.test_ds = StatefulManeuverDataset(f"{self.data_root}/test")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=False, # MANDATORY for stateful
            collate_fn=stateful_collate_fn,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, # MANDATORY for stateful
            collate_fn=stateful_collate_fn,
            num_workers=self.num_workers
        )