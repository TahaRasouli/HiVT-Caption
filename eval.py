import torch
import pytorch_lightning as pl
from datamodule import ManeuverDataModule
from model import ManeuverGRU
import glob
import os

def check_counts(dataloader):
    from collections import Counter
    counts = Counter()
    for batch in dataloader:
        _, y = batch
        targets = y.view(-1)
        # Filter out padding (-100)
        valid_targets = targets[targets != -100].tolist()
        counts.update(valid_targets)
    
    class_names = ["Maintain", "Turn L", "Turn R", "LC L", "LC R"]
    print("\n--- TEST SET CLASS DISTRIBUTION (Points) ---")
    for i, name in enumerate(class_names):
        print(f"{name:<10}: {counts[i]} points")

def run_test():
    # Performance optimization for A6000
    torch.set_float32_matmul_precision('medium')

    # Use a wildcard (*) to find the file regardless of hidden spaces or quotes
    ckpt_pattern = "/mount/arbeitsdaten65/studenten4/rasoulta/HiVT-Caption/checkpoints/maneuver-classifier-epoch=95*.ckpt"
    found_files = glob.glob(ckpt_pattern)

    if not found_files:
        print(f"❌ Error: No checkpoint matching pattern found!")
    else:
        CHECKPOINT_PATH = found_files[0]
        print(f"✅ Found checkpoint: {CHECKPOINT_PATH}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # 2. Setup DataModule
    # Note: The DataModule now automatically handles the 9-feature engineering 
    # because it uses the updated ManeuverDataset class.
    dm = ManeuverDataModule(
        data_root="/mount/studenten/projects/rasoulta/dataset/splits", 
        batch_size=1, # 1 is best for final testing to avoid padding artifacts
        num_workers=8
    )
    dm.setup(stage="test")

    check_counts(dm.test_dataloader())

    # 3. Load Model
    # We load with input_size=9 to match our new dataset structure
    model = ManeuverGRU.load_from_checkpoint(CHECKPOINT_PATH, strict=False)
    model.eval() 

    # 4. Initialize Trainer for Evaluation
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False 
    )

    print("\n" + "="*60)
    print(f"🔍 STARTING TEST EVALUATION")
    print(f"Model: {os.path.basename(CHECKPOINT_PATH)}")
    print("="*60 + "\n")

    # 5. Run Test
    trainer.test(model, dataloaders=dm.test_dataloader())

if __name__ == "__main__":
    run_test()