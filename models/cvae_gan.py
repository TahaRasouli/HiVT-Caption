import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import ADE, FDE
from models import GlobalInteractor, LocalEncoder
from models import CVAEDecoder
from models.critics import ShortScaleCritic, MidScaleCritic, LongScaleCritic
from losses import AdversarialDiscriminatorLoss, AdversarialGeneratorLoss, PhysicsLoss

class CVAE_GAN(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # Manual GAN optimization
        
        # Hyperparams
        self.historical_steps = kwargs.get("historical_steps", 20)
        self.future_steps = kwargs.get("future_steps", 30)
        embed_dim = kwargs.get("embed_dim", 128)
        self.grad_clip = kwargs.get("grad_clip", None) # Passed from train.py
        self.freeze_encoder = kwargs.get("freeze_encoder", False) # Passed from train.py
        
        # --- 1. Generator Components ---
        self.local_encoder = LocalEncoder(
            historical_steps=self.historical_steps,
            node_dim=2, edge_dim=2, embed_dim=embed_dim,
            num_heads=8, dropout=0.1, num_temporal_layers=4,
            local_radius=50
        )
        self.global_interactor = GlobalInteractor(
            historical_steps=self.historical_steps,
            embed_dim=embed_dim, edge_dim=2, 
            num_modes=1, num_heads=8, num_layers=3, dropout=0.1
        )
        self.decoder = CVAEDecoder(embed_dim=embed_dim, latent_dim=128, future_steps=self.future_steps)
        
        # --- 2. Discriminator Components ---
        self.critic_short = ShortScaleCritic(horizon=10)
        self.critic_mid = MidScaleCritic(horizon=30)
        self.critic_long = LongScaleCritic() 
        
        # --- 3. Losses & Metrics ---
        self.adv_d_loss = AdversarialDiscriminatorLoss(lambda_r1=1.0)
        self.adv_g_loss = AdversarialGeneratorLoss(lambda_adv=1.0)
        self.val_minADE = ADE() 
        self.val_minFDE = FDE()

        # --- 4. FREEZE LOGIC ---
        if self.freeze_encoder:
            print("\n[Model] FREEZING Encoder and Global Interactor (Safety Anchor).")
            for p in self.local_encoder.parameters():
                p.requires_grad = False
            for p in self.global_interactor.parameters():
                p.requires_grad = False

    def forward(self, data):
        # ... (Same as before) ...
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
        opt_g, opt_d = self.optimizers()
        
        # Prepare Data
        context = self(data)
        context = context.reshape(-1, self.hparams.embed_dim)
        y_real = data.y
        B = y_real.size(0)
        
        # Helper for slicing trajectories
        def get_multiscale_trajs(traj_tensor):
            return {'short': traj_tensor[:, :10], 'mid': traj_tensor[:, :30], 'long': traj_tensor}

        # ===================================================
        # PHASE 1: TRAIN DISCRIMINATOR
        # ===================================================
        y_fake_flat, _ = self.decoder(context, y_gt=None) 
        y_fake = y_fake_flat.reshape(B, self.future_steps, 2)
        
        real_trajs = get_multiscale_trajs(y_real)
        fake_trajs_detached = get_multiscale_trajs(y_fake.detach())
        
        loss_d, d_logs = self.adv_d_loss({
            'short': self.critic_short, 'mid': self.critic_mid, 'long': self.critic_long
        }, real_trajs, fake_trajs_detached)
        
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        
        # MANUAL GRADIENT CLIPPING FOR DISCRIMINATOR
        if self.grad_clip:
            self.clip_gradients(opt_d, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm")
            
        opt_d.step()

        # ===================================================
        # PHASE 2: TRAIN GENERATOR
        # ===================================================
        # A. Recon
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        y_recon_flat, kld_loss = self.decoder(context, data.y)
        y_recon = y_recon_flat.reshape(B, self.future_steps, 2)
        recon_loss = F.mse_loss(y_recon[reg_mask], y_real[reg_mask])
        
        # B. Adv
        y_fake_prior_flat, _ = self.decoder(context, y_gt=None)
        y_fake_prior = y_fake_prior_flat.reshape(B, self.future_steps, 2)
        fake_trajs_attached = get_multiscale_trajs(y_fake_prior)
        loss_g_adv, g_logs = self.adv_g_loss({
            'short': self.critic_short, 'mid': self.critic_mid, 'long': self.critic_long
        }, fake_trajs_attached)
        
        # C. Jerk
        loss_jerk = PhysicsLoss.compute_jerk_loss(y_fake_prior)
        
        loss_g = recon_loss + (0.01 * kld_loss) + (0.1 * loss_g_adv) + (0.05 * loss_jerk)
        
        opt_g.zero_grad()
        self.manual_backward(loss_g)

        # MANUAL GRADIENT CLIPPING FOR GENERATOR
        if self.grad_clip:
            self.clip_gradients(opt_g, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm")

        opt_g.step()

        # Log
        log_dict = {
            "train_loss_d": loss_d, "train_loss_g": loss_g,
            "train_recon": recon_loss, "train_adv": loss_g_adv, "train_jerk": loss_jerk
        }
        log_dict.update(d_logs)
        self.log_dict(log_dict, prog_bar=True, batch_size=data.num_graphs)

    # ... (Keep validation_step and on_validation_epoch_end SAME as before) ...
    @torch.no_grad()
    def validation_step(self, data, batch_idx):
        context = self(data)
        context = context.reshape(-1, self.hparams.embed_dim)
        K = 6; B = context.size(0)
        context_expanded = context.repeat_interleave(K, dim=0)
        y_hat_flat, _ = self.decoder(context_expanded, y_gt=None)
        y_hat = y_hat_flat.reshape(B, K, 30, 2).permute(1, 0, 2, 3)
        
        y_gt = data.y
        dist = torch.norm(y_hat - y_gt, p=2, dim=-1)
        ade_per_mode = dist.mean(dim=-1)
        min_ade, _ = ade_per_mode.min(dim=0)
        fde_per_mode = dist[:, :, -1]
        min_fde, _ = fde_per_mode.min(dim=0)
        
        self.log("val_minADE", min_ade.mean(), prog_bar=True, batch_size=B)
        self.log("val_minFDE", min_fde.mean(), prog_bar=True, batch_size=B)

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if self.global_rank == 0:
            print(f"\nEpoch {self.current_epoch:03d} | G Loss: {metrics.get('train_loss_g',0):.4f} | D Loss: {metrics.get('train_loss_d',0):.4f} | val_minADE: {metrics.get('val_minADE',0):.4f} | val_minFDE: {metrics.get('val_minFDE',0):.4f}")

    def configure_optimizers(self):
        # 1. GENERATOR OPTIMIZER
        # If frozen, we ONLY pass the decoder parameters
        if self.freeze_encoder:
            g_params = list(self.decoder.parameters())
        else:
            g_params = list(self.local_encoder.parameters()) + \
                       list(self.global_interactor.parameters()) + \
                       list(self.decoder.parameters())

        # 2. DISCRIMINATOR OPTIMIZER
        d_params = list(self.critic_short.parameters()) + \
                   list(self.critic_mid.parameters()) + \
                   list(self.critic_long.parameters())
        
        # Lower LR slightly for stability
        opt_g = torch.optim.AdamW(g_params, lr=1e-5, weight_decay=1e-4)
        opt_d = torch.optim.AdamW(d_params, lr=1e-5, weight_decay=1e-4)
        
        return [opt_g, opt_d], []