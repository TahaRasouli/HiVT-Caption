import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import ADE, FDE
from models import GlobalInteractor, LocalEncoder
from models import CVAEDecoder

class CVAE(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.historical_steps = kwargs.get("historical_steps", 20)
        embed_dim = kwargs.get("embed_dim", 128)
        
        # 1. Mamba Local Encoder
        self.local_encoder = LocalEncoder(
            historical_steps=self.historical_steps,
            node_dim=2, edge_dim=2, embed_dim=embed_dim,
            num_heads=8, dropout=0.1, num_temporal_layers=4,
            local_radius=50
        )
        
        # 2. Global Interactor (1 Mode)
        self.global_interactor = GlobalInteractor(
            historical_steps=self.historical_steps,
            embed_dim=embed_dim, edge_dim=2, 
            num_modes=1,  
            num_heads=8, num_layers=3, dropout=0.1
        )
        
        # 3. CVAE Decoder
        self.decoder = CVAEDecoder(embed_dim=embed_dim, latent_dim=128, future_steps=30)
        
        # 4. Metrics
        self.minADE = ADE()
        self.minFDE = FDE()
        self.val_minADE = ADE() # <--- This caused your error
        self.val_minFDE = FDE() # <--- This caused your error

    def forward(self, data):
        if self.hparams.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals; rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals; rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            data['rotate_mat'] = rotate_mat
        else:
            data['rotate_mat'] = None

        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        return global_embed

    def training_step(self, data, batch_idx):
        context = self(data)
        context = context.reshape(-1, self.hparams.embed_dim)
        
        # VARIETY LOSS LOGIC
        K = 6 
        B = context.size(0)
        
        context_expanded = context.repeat_interleave(K, dim=0)
        y_gt_expanded = data.y.repeat_interleave(K, dim=0)
        
        # Inference using Prior
        y_hat_flat, _ = self.decoder(context_expanded, y_gt=None)
        
        # Loss calculation (Winner Takes All)
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        mask_expanded = reg_mask.repeat_interleave(K, dim=0)
        
        # Compute L2 error per sample
        err = torch.norm(y_hat_flat - y_gt_expanded, p=2, dim=-1) # [B*K, Steps]
        # Mask invalid steps
        err = err * mask_expanded
        # Mean over steps
        err = err.sum(dim=-1) / (mask_expanded.sum(dim=-1) + 1e-6) # [B*K]
        
        # Find best K
        err_reshaped = err.reshape(B, K)
        min_err, _ = err_reshaped.min(dim=1)
        
        loss = min_err.mean()
        
        self.log("train_variety_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        return loss

    @torch.no_grad()
    def validation_step(self, data, batch_idx):
        context = self(data)
        context = context.reshape(-1, self.hparams.embed_dim)
        
        K = 6 
        B = context.size(0)
        context_expanded = context.repeat_interleave(K, dim=0)
        
        y_hat_flat, _ = self.decoder(context_expanded, y_gt=None)
        
        # Reshape for Metrics [Modes, Batch, Time, 2]
        y_hat = y_hat_flat.reshape(B, K, 30, 2).permute(1, 0, 2, 3)
        
        self.val_minADE.update(y_hat, data.y)
        self.val_minFDE.update(y_hat, data.y)
        
        self.log("val_minADE", self.val_minADE, prog_bar=True, batch_size=B)
        self.log("val_minFDE", self.val_minFDE, prog_bar=True, batch_size=B)

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if self.global_rank == 0:
            ade = metrics.get('val_minADE', 0.0)
            fde = metrics.get('val_minFDE', 0.0)
            loss = metrics.get('train_variety_loss', 0.0)
            print(f"\nEpoch {self.current_epoch:03d} | Variety Loss: {loss:.4f} | val_minADE: {ade:.4f} | val_minFDE: {fde:.4f}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-4, weight_decay=1e-4)