import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PytorchMambaBlock(nn.Module):
    """
    A lightweight, pure-PyTorch implementation of the Mamba Block.
    It mimics the architecture: Input -> Expand -> Conv1d -> SSM -> Gating -> Output.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv
        self.d_state = d_state
        
        # 1. Expansion Projector
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 2. 1D Convolution (Simulates local context)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # 3. Activation
        self.act = nn.SiLU()
        
        # 4. Simplified SSM Parameters (Learnable A and D)
        # A: (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(torch.randn(self.d_inner, d_state).abs()))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Discretization step (delta)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # 5. Output Projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Input x: [Batch, Time, Dim]
        """
        B, L, D = x.shape
        
        # 1. Project to higher dim (x and z for gating)
        xz = self.in_proj(x) # [B, L, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1) # Split into signal (x) and gate (z)
        
        # 2. Conv1d (needs [B, Dim, Time])
        x_conv = x_proj.permute(0, 2, 1)
        x_conv = self.conv1d(x_conv)[:, :, :L] # Causal padding trick
        x_conv = x_conv.permute(0, 2, 1)
        
        # 3. Activation
        x_act = self.act(x_conv)
        
        # 4. SSM Scan (The "Mamba" part)
        # Since L=20 (short history), we can implement a simplified scan loop
        # y_t = A * y_{t-1} + x_t
        
        # Compute dynamic step size 'delta'
        dt = F.softplus(self.dt_proj(x_act)) # [B, L, d_inner]
        
        # Discretize A (Simplified Zero-Order Hold)
        A = -torch.exp(self.A_log) # [d_inner, d_state]
        # We approximate the scan for speed using a cumulative sum/product logic 
        # OR a simple recurrent loop since L is small.
        
        # -- Simplified Recurrence for Thesis Stability --
        # This acts like the selective scan but is pure torch
        scan_output = []
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        
        for t in range(L):
            # x_t: [B, d_inner]
            xt = x_act[:, t, :]
            dt_t = dt[:, t, :] # [B, d_inner]
            
            # Discretize A and B for this step
            # dA = exp(delta * A)
            dA = torch.exp(torch.einsum('bi,is->bis', dt_t, A)) # [B, d_inner, d_state]
            dB = dt_t.unsqueeze(-1) * xt.unsqueeze(-1) # [B, d_inner, 1] * [B, d_inner, 1] approx
            
            # State Update: h_t = dA * h_{t-1} + dB
            h = dA * h + dB 
            
            # Output: y_t = h_t * 1 + D * x_t
            y_t = h.sum(dim=-1) + self.D * xt
            scan_output.append(y_t)
            
        y_ssm = torch.stack(scan_output, dim=1) # [B, L, d_inner]
        
        # 5. Gating
        y_gated = y_ssm * self.act(z)
        
        # 6. Output Project
        out = self.out_proj(y_gated)
        return out

class MambaTemporalEncoder(nn.Module):
    """
    Drop-in replacement for HiVT's TemporalEncoder.
    Uses the PytorchMambaBlock.
    """
    def __init__(self, embed_dim=128, historical_steps=20, num_layers=2):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PytorchMambaBlock(d_model=embed_dim),
                nn.LayerNorm(embed_dim)
            ]) for _ in range(num_layers)
        ])
        
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, padding_mask=None):
        """
        Input x: [Time, Batch, Embed] (HiVT format)
        Output: [Batch, Embed] (Latent representation of the history)
        """
        # 1. Permute to [Batch, Time, Embed] for Mamba processing
        x = x.permute(1, 0, 2) 
        
        # 2. Run Mamba Layers
        for mamba_layer, norm_layer in self.layers:
            res = x
            x = mamba_layer(x)
            x = norm_layer(x + res)
            
        # 3. Permute back to [Time, Batch, Embed]
        x = x.permute(1, 0, 2)
        
        # 4. Normalize
        x = self.out_norm(x)
        
        # --- THE FIX IS HERE ---
        # The original TemporalEncoder returns 'out[-1]' (The last hidden state).
        # We must do the same to collapse the Time dimension.
        return x[-1]  # Shape becomes [Batch, Embed]