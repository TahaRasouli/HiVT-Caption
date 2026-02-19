import os
import glob
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from datamodule import ManeuverDataModule
from model import ManeuverGRU

def train():
    # 0. SETUP PATHS - Using your absolute path
    # Ensure this directory exists
    project_root = "/mount/arbeitsdaten/studenten4/rasoulta/HiVT-Caption/"
    ckpt_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Performance optimization for A6000 Tensor Cores
    torch.set_float32_matmul_precision('medium')
    
    # 1. INITIALIZE DATA AND MODEL
    dm = ManeuverDataModule(
        data_root="/mount/studenten/projects/rasoulta/dataset/splits", 
        batch_size=16,
        num_workers=16 
    )
    
    maneuver_weights = [1.00, 3.91, 3.90, 4.33, 5.52]
    
    model = ManeuverGRU(
        input_size=6, 
        hidden_size=128, 
        num_classes=5, 
        lr=1e-3, 
        weights=maneuver_weights
    )

    # 2. CONFIGURE CHECKPOINT CALLBACK
    # Saving Top 5 based on Macro F1 (Higher is better)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1_macro", 
        dirpath=ckpt_dir,
        filename="maneuver-classifier-{epoch:02d}-{val_f1_macro:.4f}",
        save_top_k=5,
        mode="max",
        save_last=True # Safety net: saves last.ckpt regardless of score
    )

    # 3. TRAINER
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
        default_root_dir=project_root # Puts logs and temp files in your project root
    )

    # 4. START TRAINING
    print(f"🚀 Training starting. Checkpoints will be saved to: {ckpt_dir}")
    trainer.fit(model, dm)

    # 5. POST-TRAINING VERIFICATION
    print("\n" + "="*60)
    print("🏆 TRAINING COMPLETE: SAVED CHECKPOINTS")
    print("="*60)

    # Find and sort the actual files on disk
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    
    def extract_f1(path):
        try:
            # Parses '0.8703' from '...val_f1_macro=0.8703.ckpt'
            return float(path.split("val_f1_macro=")[-1].replace(".ckpt", ""))
        except:
            return 0.0

    # Sort files by the F1 score in the filename (descending)
    performance_ckpts = [f for f in ckpt_files if "last" not in f]
    performance_ckpts.sort(key=extract_f1, reverse=True)

    if not performance_ckpts:
        print(f"⚠️ Warning: No checkpoint files found in {ckpt_dir}!")
    else:
        print(f"{'Rank':<5} | {'F1 Macro':<10} | {'Filename'}")
        print("-" * 60)
        for i, ckpt in enumerate(performance_ckpts[:5]):
            f1_val = extract_f1(ckpt)
            print(f"{i+1:<5} | {f1_val:<10.4f} | {os.path.basename(ckpt)}")
    
    print("="*60)
    print(f"Best model according to Lightning: {checkpoint_callback.best_model_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    train()