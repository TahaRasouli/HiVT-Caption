from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import torch.multiprocessing as mp
import torch
import torch.utils._pytree
# Monkey-patch to fix transformers/torch version mismatch
if not hasattr(torch.utils._pytree, 'register_pytree_node'):
    torch.utils._pytree.register_pytree_node = torch.utils._pytree._register_pytree_node

from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.hivt import HiVT

# speed boost on Nvidia-A6000
torch.set_float32_matmul_precision('medium')
mp.set_start_method('spawn', force=True)

def main():
    pl.seed_everything(2022)
    parser = ArgumentParser()

    # Data arguments
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--persistent_workers", type=bool, default=False)
    parser.add_argument("--ckpt_path", type=str, default=None)

    # Training arguments
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=64)
    parser.add_argument("--monitor", type=str, default="val_minFDE", choices=["val_minADE", "val_minFDE", "val_minMR"])
    parser.add_argument("--save_top_k", type=int, default=5)

    # HiVT model specific args
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    # 1. Lower the Learning Rate for fine-tuning if a checkpoint is provided
    if args.ckpt_path:
        print(f"Fine-tuning detected. Lowering Learning Rate to 1e-4")
        args.lr = 1e-4 

    # 2. Model Initialization
    model = HiVT(**vars(args))

    # --- WARM START LOGIC (Improved for Transfer Learning) ---
    actual_fit_path = args.ckpt_path
    if args.ckpt_path:
        print(f"--- Loading Weights from: {args.ckpt_path} ---")
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        state_dict = ckpt['state_dict']
        
        # Check for GAN critics to decide if we are resuming or transferring
        has_critics = any("D_short" in k for k in state_dict.keys())
        
        if not has_critics:
            print("Detected Supervised/CVAE checkpoint. performing SMART TRANSFER.")
            
            # 1. Filter out incompatible layers (Decoder & Multi-Head Projections)
            keys_to_remove = []
            for key in state_dict.keys():
                # Remove Decoder (CVAE decoder != GAN decoder)
                if "decoder" in key:
                    keys_to_remove.append(key)
                # Remove Global Interactor Output Head (1 mode != 6 modes)
                if "global_interactor.multihead_proj" in key:
                    keys_to_remove.append(key)
            
            # 2. Actually delete them
            for k in keys_to_remove:
                print(f"Dropping mismatched key: {k}")
                del state_dict[k]
                
            # 3. Load the rest (Encoders + Global Blocks)
            model.load_state_dict(state_dict, strict=False)
            
            # Reset actual_fit_path so we start a NEW training run (epoch 0)
            actual_fit_path = None 
        else:
            print("Detected GAN checkpoint. Full resume enabled.")
            # If it's a GAN checkpoint, we assume shapes match exactly.

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,
        save_top_k=args.save_top_k,
        mode="min",
    )

    # Trainer
    strategy = DDPStrategy(find_unused_parameters=True)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy=strategy,
        precision="16-mixed",  
        # gradient_clip_val=0.5,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback], # Critical to include this
        log_every_n_steps=50,
        num_sanity_val_steps=0
    )

    datamodule = NuScenesHiVTDataModule(
        root=args.root,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    trainer.fit(model, datamodule, ckpt_path=actual_fit_path)

if __name__ == "__main__":
    main()