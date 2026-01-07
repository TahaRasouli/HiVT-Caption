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
        
        # Encoders
        self.local_encoder = LocalEncoder(
            historical_steps=self.historical_steps,
            node_dim=2, edge_dim=2, embed_dim=embed_dim,
            num_heads=8, dropout=0.1, num_temporal_layers=4,
            local_radius=50
        )
        # Force num_modes=1 for CVAE (Diversity comes from z, not heads)
        self.global_interactor = GlobalInteractor(
            historical_steps=self.historical_steps,
            embed_dim=embed_dim, edge_dim=2, 
            num_modes=1,  # <--- IMPORTANT
            num_heads=8, num_layers=3, dropout=0.1
        )
        
        # Decoder
        self.decoder = CVAEDecoder(embed_dim=embed_dim, latent_dim=16, future_steps=30)
        
        self.minADE = ADE()
        self.minFDE = FDE()


    def forward(self, data):
        # Rotation logic
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
        # Reshape [1, 1, N, 128] -> [N, 128]
        context = context.reshape(-1, self.hparams.embed_dim)
        
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        
        # Forward Pass with GT (Training Mode)
        y_hat, kld_loss = self.decoder(context, data.y)
        
        # Reconstruction Loss (L2 or Huber)
        # Only calc loss on valid steps
        recon_loss = F.mse_loss(y_hat[reg_mask], data.y[reg_mask])
        
        # Total Loss = Recon + lambda * KLD
        # usually lambda starts small and anneals, but 0.05 is a safe start
        loss = recon_loss + 0.01 * kld_loss 
        
        self.log("train_recon_loss", recon_loss, prog_bar=True, batch_size=data.num_graphs)
        self.log("train_kld_loss", kld_loss, prog_bar=True, batch_size=data.num_graphs)
        return loss

    @torch.no_grad()
    def validation_step(self, data, batch_idx):
        context = self(data)
        context = context.reshape(-1, self.hparams.embed_dim)
        
        # Generate Multiple Modes (e.g., 6 samples)
        # We simulate "Modes" by sampling z 6 times
        K = 6 
        B = context.size(0)
        
        # Repeat context K times: [B*K, Embed]
        context_expanded = context.repeat_interleave(K, dim=0)
        
        # Inference (No GT provided)
        y_hat_flat, _ = self.decoder(context_expanded, y_gt=None)
        
        # Reshape to [B, K, 30, 2] -> [Batch, Modes, Time, 2] for Metrics
        y_hat = y_hat_flat.reshape(B, K, 30, 2)
        y_hat = y_hat.permute(1, 0, 2, 3) # [Modes, Batch, Time, 2]
        
        # Metrics expects [Modes, Batch, Time, 2]
        self.minADE.update(y_hat, data.y)
        self.minFDE.update(y_hat, data.y)
        
        self.log("val_minFDE", self.val_minFDE, prog_bar=True, batch_size=B)

    def on_validation_epoch_end(self):
        """
        Prints a clean summary of the epoch's performance to the terminal.
        Includes CVAE-specific losses (Recon + KLD) to track latent learning.
        """
        metrics = self.trainer.callback_metrics
        if self.global_rank == 0:
            # Fetch metrics from the trainer's dictionary
            ade = metrics.get('val_minADE', 0.0)
            fde = metrics.get('val_minFDE', 0.0)
            recon = metrics.get('train_recon_loss', 0.0)
            kld = metrics.get('train_kld_loss', 0.0)
            
            print(f"\nEpoch {self.current_epoch:03d} | "
                  f"Recon Loss: {recon:.4f} | "
                  f"KLD Loss: {kld:.4f} | "
                  f"val_minADE: {ade:.4f} | "
                  f"val_minFDE: {fde:.4f}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-4, weight_decay=1e-4)
