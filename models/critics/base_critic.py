import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseTrajectoryCritic(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(4) 
        )

        dim = 128 * 4
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        vel = traj[:, 1:] - traj[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]
        
        kinematics = torch.cat([vel[:, :-1], acc], dim=-1).transpose(1, 2)
        
        # Tanh completely protects the CNN from extreme velocity spikes
        kinematics = torch.tanh(kinematics)
        
        features = self.conv_block(kinematics)
        flat = features.view(features.size(0), -1)
        
        raw_out = self.out(self.mlp(flat)).squeeze(-1)
        
        # Bounding the Critic output guarantees MSELoss cannot explode
        return torch.tanh(raw_out) * 10.0