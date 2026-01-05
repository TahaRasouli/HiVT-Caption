import os
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.hivt import HiVT

def evaluate():
    parser = ArgumentParser()
    
    # Required arguments
    parser.add_argument("--root", type=str, required=True, help="Path to processed dataset root")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the .ckpt file")
    
    # Optional performance arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--devices", type=int, default=1)
    
    # Model specific args (to ensure dimensions match)
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    print(f"--- Loading Model from Checkpoint: {args.ckpt_path} ---")
    
    # 1. Load the model
    # strict=False allows loading even if some GAN components are missing in the ckpt
    model = HiVT.load_from_checkpoint(args.ckpt_path, strict=False, **vars(args))
    model.eval()

    # 2. Initialize DataModule
    datamodule = NuScenesHiVTDataModule(
        root=args.root,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False
    )

    # 3. Initialize Trainer for Validation
    # We use 16-mixed to match your training speed/precision
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        precision="16-mixed",
        logger=False # We only want the console output
    )

    print("\n--- Starting Validation Epoch ---")
    
    # 4. Run Validation
    # This calls the validation_step in your hivt.py
    results = trainer.validate(model, datamodule=datamodule, verbose=True)

    # 5. Formatted Print
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS (NuScenes Validation Set)")
    print("="*50)
    
    res = results[0]
    metrics_to_print = [
        ("MinADE", "val_minADE"),
        ("MinFDE", "val_minFDE"),
        ("Miss Rate", "val_minMR"),
        ("Jerk", "val_jerk"),
        ("Speed Violation", "val_speed"),
        ("Diversity", "val_div")
    ]

    for label, key in metrics_to_print:
        val = res.get(key, "N/A")
        if isinstance(val, float):
            print(f" • {label:18}: {val:.4f}")
        else:
            print(f" • {label:18}: {val}")
    
    print("="*50)

if __name__ == "__main__":
    evaluate()
