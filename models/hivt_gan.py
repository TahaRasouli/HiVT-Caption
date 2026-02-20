import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import ADE, FDE
from models import GlobalInteractor, LocalEncoder
from models.decoder import MLPDecoder  # Ensure your MLPDecoder is in this file

from models.critics import ShortScaleCritic, MidScaleCritic, LongScaleCritic
from losses import AdversarialDiscriminatorLoss, AdversarialGeneratorLoss
from losses import PhysicsLoss

class HiVTGAN(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.historical_steps = kwargs.get("historical_steps", 20)
        embed_dim = kwargs.get("embed_dim", 128)
        
        self.local_encoder = LocalEncoder(
            historical_steps=self.historical_steps,
            node_dim=2, edge_dim=2, embed_dim=embed_dim,
            num_heads=8, dropout=0.1, num_temporal_layers=4,
            local_radius=50
        )
        self.global_interactor = GlobalInteractor(
            historical_steps=self.historical_steps,
            embed_dim=embed_dim, edge_dim=2, 
            num_modes=6,  # <--- CRITICAL FIX: Change this from 1 to 6
            num_heads=8, num_layers=3, dropout=0.1
        )

        # 2. MLP Decoder (Replaces CVAE)
        self.decoder = MLPDecoder(
            local_channels=embed_dim,
            global_channels=embed_dim,
            future_steps=30,
            num_modes=6,
            uncertain=False # Set to False to output purely [B, T, 2] for the critics
        )
        
        # 3. Discriminator Components
        self.critics = nn.ModuleDict({
            'short': ShortScaleCritic(horizon=10),
            'mid': MidScaleCritic(horizon=30),
            'long': LongScaleCritic()
        })
        
        # 4. Losses
        self.d_loss_fn = AdversarialDiscriminatorLoss(lambda_r1=1.0)
        self.g_loss_fn = AdversarialGeneratorLoss(lambda_adv=1.0)
        self.lambda_adv = 0.1 
        self.lambda_jerk = 0.05
        
        # 5. Metrics
        self.val_minADE = ADE() 
        self.val_minFDE = FDE() 

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
        
        # Return both for the original decoder
        return local_embed, global_embed

    def training_step(self, data, batch_idx):
        opt_g, opt_d = self.optimizers()
        
        # ===============================
        # 1. GENERATE FAKE TRAJECTORIES
        # ===============================
        local_embed, global_embed = self(data)
        
        # loc shape: [6, B, 30, 2] | pi shape: [B, 6]
        loc, pi = self.decoder(local_embed, global_embed)
        
        # Permute to [B, 6, 30, 2] for easier processing
        y_hat_reshaped = loc.permute(1, 0, 2, 3) 
        B, K, H, D = y_hat_reshaped.shape
        y_gt = data.y
        
        # Masking and Variety Loss
        reg_mask = ~data['padding_mask'][:, self.historical_steps:] # [B, 30]
        mask_expanded = reg_mask.unsqueeze(1).expand(B, K, H)       # [B, 6, 30]
        y_gt_expanded = y_gt.unsqueeze(1).expand(B, K, H, D)        # [B, 6, 30, 2]
        
        err = torch.norm(y_hat_reshaped - y_gt_expanded, p=2, dim=-1) * mask_expanded
        err = err.sum(dim=-1) / (mask_expanded.sum(dim=-1) + 1e-6)  # [B, 6]
        
        # Find best mode
        min_err, best_idx = err.min(dim=1)
        variety_loss = min_err.mean()
        
        # Probability Loss (Teach 'pi' to predict the best index)
        pi_loss = F.cross_entropy(pi, best_idx)
        
        # Select best fakes for the Discriminator
        best_fakes = y_hat_reshaped[torch.arange(B), best_idx] # [B, 30, 2]
        
        real_dict = {'short': y_gt, 'mid': y_gt, 'long': y_gt}
        fake_dict = {'short': best_fakes, 'mid': best_fakes, 'long': best_fakes}

        # ===============================
        # 2. TRAIN DISCRIMINATOR
        # ===============================
        self.toggle_optimizer(opt_d)
        opt_d.zero_grad()
        
        detached_fakes = {k: v.detach() for k, v in fake_dict.items()}
        d_loss, d_logs = self.d_loss_fn(self.critics, real_dict, detached_fakes)
        
        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # ===============================
        # 3. TRAIN GENERATOR
        # ===============================
        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()
        
        g_adv_loss, g_logs = self.g_loss_fn(self.critics, fake_dict)
        jerk_loss = PhysicsLoss.compute_jerk_loss(best_fakes)
        
        # Total Generator Loss
        g_total_loss = variety_loss + pi_loss + (self.lambda_adv * g_adv_loss) + (self.lambda_jerk * jerk_loss)
        
        self.manual_backward(g_total_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        # ===============================
        # 4. LOGGING
        # ===============================
        self.log("train_variety_loss", variety_loss, prog_bar=True, batch_size=B)
        self.log("train_pi_loss", pi_loss, prog_bar=False, batch_size=B)
        self.log("g_loss", g_total_loss, prog_bar=False, batch_size=B)
        self.log("d_loss", d_loss, prog_bar=False, batch_size=B)
        for k, v in {**d_logs, **g_logs}.items():
            self.log(k, v, prog_bar=False, batch_size=B)

    @torch.no_grad()
    def validation_step(self, data, batch_idx):
        local_embed, global_embed = self(data)
        loc, pi = self.decoder(local_embed, global_embed)
        
        # loc shape: [6, B, 30, 2] -> permute to [B, 6, 30, 2]
        y_hat = loc.permute(1, 0, 2, 3) 
        y_gt = data.y 
        B = y_hat.size(0)
        
        dist = torch.norm(y_hat - y_gt.unsqueeze(1), p=2, dim=-1)
        
        ade_per_mode = dist.mean(dim=-1)
        min_ade_per_agent, _ = ade_per_mode.min(dim=1)
        val_minADE = min_ade_per_agent.mean()

        fde_per_mode = dist[:, :, -1]
        min_fde_per_agent, _ = fde_per_mode.min(dim=1)
        val_minFDE = min_fde_per_agent.mean()
        
        self.log("val_minADE", val_minADE, prog_bar=True, batch_size=B)
        self.log("val_minFDE", val_minFDE, prog_bar=True, batch_size=B)

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if self.global_rank == 0:
            ade = metrics.get('val_minADE', 0.0)
            fde = metrics.get('val_minFDE', 0.0)
            print(f"\nEpoch {self.current_epoch:03d} | val_minADE: {ade:.4f} | val_minFDE: {fde:.4f}")

    def configure_optimizers(self):
        g_params = list(self.local_encoder.parameters()) + \
                   list(self.global_interactor.parameters()) + \
                   list(self.decoder.parameters())
        d_params = self.critics.parameters()
        
        opt_g = torch.optim.AdamW(g_params, lr=self.hparams.lr, weight_decay=1e-4)
        opt_d = torch.optim.AdamW(d_params, lr=self.hparams.lr * 0.5, weight_decay=1e-4)
        
        return [opt_g, opt_d], []