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
        self.d_loss_fn = AdversarialDiscriminatorLoss(lambda_r1=0.0)
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
        # 1. IMMEDIATE SANITIZATION
        # ===============================
        # We MUST clean the data before it enters the model or hits the rotate_mat
        if data.y is not None:
            data.y = torch.nan_to_num(data.y, nan=0.0, posinf=0.0, neginf=0.0)
            data.y = torch.clamp(data.y, min=-100.0, max=100.0)
            
        local_embed, global_embed = self(data)
        loc, pi = self.decoder(local_embed, global_embed)
        
        if torch.isnan(loc).any():
            raise ValueError(f"FATAL: Generator outputted NaN at Step {batch_idx}.")
            
        y_hat_reshaped = loc.permute(1, 0, 2, 3) 
        B, K, H, D = y_hat_reshaped.shape
        y_gt = data.y
        
        # ===============================
        # 2. ISOLATE VALID AGENTS
        # ===============================
        valid_mask = ~data['padding_mask'][:, self.historical_steps:] # [B, 30]
        agent_has_future = valid_mask.any(dim=-1) # [B]
        
        # DDP Failsafe for empty batches
        is_empty_batch = not agent_has_future.any()
        if is_empty_batch:
            y_gt_valid = torch.zeros((1, H, D), device=self.device, requires_grad=True)
            best_fakes_valid = torch.zeros((1, H, D), device=self.device, requires_grad=True)
            pi_valid = torch.zeros((1, K), device=self.device, requires_grad=True)
            mask_valid = torch.ones((1, H), dtype=torch.bool, device=self.device)
            N = 1
        else:
            y_hat_valid = y_hat_reshaped[agent_has_future] # [N, 6, 30, 2]
            y_gt_valid = y_gt[agent_has_future] # [N, 30, 2]
            mask_valid = valid_mask[agent_has_future] # [N, 30]
            pi_valid = pi[agent_has_future] # [N, 6]
            N = y_hat_valid.size(0)

        # ===============================
        # 3. GENERATOR LOSS (Valid Agents Only)
        # ===============================
        if not is_empty_batch:
            mask_expanded = mask_valid.unsqueeze(1).expand(N, K, H)      
            y_gt_expanded = y_gt_valid.unsqueeze(1).expand(N, K, H, D)         
            
            diff = y_hat_valid - y_gt_expanded
            err = torch.sqrt(diff.pow(2).sum(dim=-1) + 1e-6) * mask_expanded
            
            # Since these are valid agents, mask_expanded.sum > 0, preventing 1e6 gradient spikes
            err = err.sum(dim=-1) / (mask_expanded.sum(dim=-1) + 1e-6)  
            
            min_err, best_idx = err.min(dim=1)
            variety_loss = min_err.mean()
            pi_loss = F.cross_entropy(pi_valid, best_idx)
            
            best_fakes_valid = y_hat_valid[torch.arange(N), best_idx] # [N, 30, 2]
        else:
            variety_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            pi_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        real_dict = {'short': y_gt_valid, 'mid': y_gt_valid, 'long': y_gt_valid}
        fake_dict = {'short': best_fakes_valid, 'mid': best_fakes_valid, 'long': best_fakes_valid}

        # ===============================
        # 4. TRAIN DISCRIMINATOR
        # ===============================
        self.toggle_optimizer(opt_d)
        opt_d.zero_grad()
        
        detached_fakes = {k: v.detach() for k, v in fake_dict.items()}
        d_loss, d_logs = self.d_loss_fn(self.critics, real_dict, detached_fakes)
        
        if is_empty_batch: d_loss = d_loss * 0.0 
            
        self.manual_backward(d_loss)
        
        # --- THE ULTIMATE FAILSAFE: GRADIENT SCRUBBING ---
        for p in self.critics.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), max_norm=2.0)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # ===============================
        # 5. TRAIN GENERATOR
        # ===============================
        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()
        
        g_adv_loss, g_logs = self.g_loss_fn(self.critics, fake_dict)
        jerk_loss = PhysicsLoss.compute_jerk_loss(best_fakes_valid)
        
        g_total_loss = variety_loss + pi_loss + (self.lambda_adv * g_adv_loss) + (self.lambda_jerk * jerk_loss)
        
        if is_empty_batch: g_total_loss = g_total_loss * 0.0
            
        self.manual_backward(g_total_loss)
        
        g_params = list(self.local_encoder.parameters()) + list(self.global_interactor.parameters()) + list(self.decoder.parameters())
        
        # --- THE ULTIMATE FAILSAFE: GRADIENT SCRUBBING ---
        for p in g_params:
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                
        torch.nn.utils.clip_grad_norm_(g_params, max_norm=2.0)
        
        opt_g.step()
        self.untoggle_optimizer(opt_g)
        
        # ===============================
        # 6. LOGGING
        # ===============================
        if not is_empty_batch:
            self.log("train_variety_loss", variety_loss, prog_bar=True, batch_size=B)
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