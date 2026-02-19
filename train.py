import os
import glob
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from datamodule import ManeuverDataModule
from model import StatefulManeuverGRU

def train():
    # 0. SETUP PATHS
    # Using your requested absolute path
    project_root = "/mount/arbeitsdaten/studenten4/rasoulta/HiVT-Caption/"
    ckpt_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Performance optimization for A6000 Tensor Cores
    torch.set_float32_matmul_precision('medium')
    
    # 1. INITIALIZE DATA AND MODEL
    # IMPORTANT: Ensure your DataModule/DataLoader sets shuffle=False and batch_size=1
    dm = ManeuverDataModule(
        data_root="/mount/studenten/projects/rasoulta/dataset/splits", 
        batch_size=1,      # REQUIRED for simple stateful state passing
        num_workers=8      # Slightly lower workers often helps with sequential disk reads
    )
    
    # Maneuver weights from your previous point distribution analysis
    maneuver_weights = [1.00, 3.91, 3.90, 4.33, 5.52]
    
    model = StatefulManeuverGRU(
        input_size=6, 
        hidden_size=128, 
        num_layers=2, 
        num_classes=5, 
        lr=1e-4,           # Lower learning rate often helps stateful models stabilize
        weights=maneuver_weights
    )

    # 2. CONFIGURE CHECKPOINT CALLBACK
    # Saving Top 5 based on Macro F1 (the best balance of all maneuvers)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1_macro", 
        dirpath=ckpt_dir,
        filename="stateful-maneuver-{epoch:02d}-{val_f1_macro:.4f}",
        save_top_k=5,
        mode="max",
        save_last=True
    )

    # 3. TRAINER
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        default_root_dir=project_root
    )

    # 4. START TRAINING
    print(f"🚀 Stateful training starting. Checkpoints: {ckpt_dir}")
    print("⚠️  Note: batch_size=1 and shuffle=False are active for temporal context.")
    trainer.fit(model, dm)

    # 5. POST-TRAINING: Print the best 5 saved files
    print("\n" + "="*60)
    print("🏆 STATEFUL TRAINING COMPLETE: TOP 5 CHECKPOINTS")
    print("="*60)

    ckpt_files = glob.glob(os.path.join(ckpt_dir, "stateful-maneuver-*.ckpt"))
    
    def extract_f1(path):
        try:
            return float(path.split("val_f1_macro=")[-1].replace(".ckpt", ""))
        except:
            return 0.0

    performance_ckpts = [f for f in ckpt_files if "last" not in f]
    performance_ckpts.sort(key=extract_f1, reverse=True)

    if not performance_ckpts:
        print(f"⚠️ No checkpoints found in {ckpt_dir}!")
    else:
        print(f"{'Rank':<5} | {'F1 Macro':<10} | {'Filename'}")
        print("-" * 60)
        for i, ckpt in enumerate(performance_ckpts[:5]):
            print(f"{i+1:<5} | {extract_f1(ckpt):<10.4f} | {os.path.basename(ckpt)}")
    
    print("="*60)
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    train()