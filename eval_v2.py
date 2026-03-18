import torch
import pytorch_lightning as pl
from datamodule import ManeuverDataModule
from model import ManeuverGRU
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
    print("\n" + "="*40)
    print("📊 TEST SET CLASS DISTRIBUTION (Points)")
    print("="*40)
    for i, name in enumerate(class_names):
        print(f"{name:<15}: {counts[i]} points")
    print("="*40 + "\n")

def run_test():
    # Performance optimization for A6000
    torch.set_float32_matmul_precision('medium')

    # 1. PATH TO YOUR BEST CHECKPOINT
    CHECKPOINT_PATH = "/mount/arbeitsdaten/studenten4/rasoulta/HiVT-Caption/checkpoints/maneuver-classifier-epoch=36-val_f1_macro=0.9223.ckpt"

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # 2. Setup DataModule
    dm = ManeuverDataModule(
        data_root="/mount/studenten/projects/rasoulta/dataset/splits", 
        batch_size=1, # 1 is best for final testing to avoid padding artifacts
        num_workers=8
    )
    dm.setup(stage="test")
    
    # Instantiate the dataloader once to save memory/overhead
    test_loader = dm.test_dataloader()

    # Verify what's actually in the test set
    check_counts(test_loader)

    # 3. Load Model
    # We must explicitly pass the weights here because we ignored them in save_hyperparameters
    maneuver_weights = [1.00, 3.91, 3.90, 4.33, 5.52]
    
    model = ManeuverGRU.load_from_checkpoint(
        CHECKPOINT_PATH, 
        weights=maneuver_weights
    )
    model.eval()
    
    # Note: trainer.test() automatically calls model.eval() and torch.no_grad() under the hood!

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
    trainer.test(model, dataloaders=test_loader)

if __name__ == "__main__":
    run_test()
