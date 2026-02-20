import os
import glob
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from datamodule import ManeuverDataModule
from model import ManeuverCNN 


def train():
    # 0. REPRODUCIBILITY & PERFORMANCE
    # Fixing the seed is vital to compare CNN vs GRU performance fairly
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    # Absolute path to your workspace
    project_root = "/mount/arbeitsdaten65/studenten4/rasoulta/HiVT-Caption/"
    ckpt_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. INITIALIZE DATA
    # Batch size 32 works well for CNNs to maintain stable BatchNorm statistics
    dm = ManeuverDataModule(
        data_root="/mount/studenten/projects/rasoulta/dataset/splits", 
        batch_size=16,      
        num_workers=8      
    )
    
    # 2. STRATEGIC CLASS WEIGHTS
    # Based on your previous counts, we keep the weights that helped Turn L
    maneuver_weights = [1.00, 4.50, 3.90, 4.33, 5.52]
    
    # 3. INITIALIZE CNN MODEL
    model = ManeuverCNN(
        input_size=9,      # 6 raw features + 3 deltas
        num_classes=5, 
        lr=1e-3,           
        weights=maneuver_weights
    )

    # 4. CALLBACKS
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1_macro", 
        dirpath=ckpt_dir,
        filename="cnn-maneuver-{epoch:02d}-{val_f1_macro:.4f}",
        save_top_k=5,
        mode="max",
        save_last=True
    )

    # CNNs usually converge faster than RNNs, but can plateau early
    early_stop_callback = EarlyStopping(
        monitor="val_f1_macro",
        patience=25,
        mode="max"
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 5. TRAINER
    trainer = pl.Trainer(
        max_epochs=100,      
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=5,
        deterministic=True, # Critical for F1 research consistency
        default_root_dir=project_root
    )

    # 6. START TRAINING
    print(f"🚀 Starting 1D-CNN Training. Let's see if we can beat 0.90 F1!")
    trainer.fit(model, dm)

    # 7. POST-TRAINING SUMMARY
    print("\n" + "="*60)
    print("🏆 TRAINING COMPLETE: TOP 5 CNN CHECKPOINTS")
    print("="*60)

    ckpt_files = glob.glob(os.path.join(ckpt_dir, "cnn-maneuver-*.ckpt"))
    
    def extract_f1(path):
        try:
            return float(path.split("val_f1_macro=")[-1].replace(".ckpt", ""))
        except:
            return 0.0

    performance_ckpts = [f for f in ckpt_files if "last" not in f]
    performance_ckpts.sort(key=extract_f1, reverse=True)

    if not performance_ckpts:
        print(f"⚠️ No CNN checkpoints found in {ckpt_dir}!")
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