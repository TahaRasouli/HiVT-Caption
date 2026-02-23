import torch
import torch.nn as nn

class AdversarialDiscriminatorLoss(nn.Module):
    def __init__(self, lambda_r1: float = 0.0):
        super(AdversarialDiscriminatorLoss, self).__init__()
        self.lambda_r1 = lambda_r1
        self.mse = nn.MSELoss() # LSGAN is mathematically safe

    def forward(self, critics: dict, real_trajs: dict, fake_trajs: dict):
        total_loss = 0.0
        logs = {}

        for scale in critics.keys():
            d_real = critics[scale](real_trajs[scale].detach())
            d_fake = critics[scale](fake_trajs[scale].detach())

            # LSGAN: Real -> 1, Fake -> 0
            real_loss = self.mse(d_real, torch.ones_like(d_real))
            fake_loss = self.mse(d_fake, torch.zeros_like(d_fake))
            
            # Standard LSGAN averaging
            scale_loss = 0.5 * (real_loss + fake_loss)
            total_loss += scale_loss
            
            logs[f"d_loss_{scale}"] = scale_loss.detach()
            logs[f"d_r1_{scale}"] = torch.tensor(0.0) # Disabled to prevent DDP bugs

        return total_loss, logs

class AdversarialGeneratorLoss(nn.Module):
    def __init__(self, lambda_adv: float = 1.0):
        super(AdversarialGeneratorLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.mse = nn.MSELoss()

    def forward(self, critics: dict, fake_trajs: dict):
        total_loss = 0.0
        logs = {}

        for scale in critics.keys():
            d_fake = critics[scale](fake_trajs[scale])
            
            # Generator wants the Discriminator to output 1
            scale_loss = self.mse(d_fake, torch.ones_like(d_fake))
            total_loss += scale_loss
            logs[f"g_loss_{scale}"] = scale_loss.detach()

        return total_loss, logs