import torch
import torch.nn as nn


class R1Regularization(nn.Module):
    def __init__(self):
        super(R1Regularization, self).__init__()

    def forward(self, d_out: torch.Tensor, real_data: torch.Tensor) -> torch.Tensor:
        grad_real = torch.autograd.grad(
            outputs=d_out.sum(),
            inputs=real_data,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # --- NEW: CRITICAL SAFETY CLAMP ---
        # Prevent the gradients from exploding before they are squared
        grad_real = torch.clamp(grad_real, min=-10.0, max=10.0)
        
        grad_penalty = grad_real.pow(2).reshape(grad_real.size(0), -1).sum(dim=1)
        return grad_penalty.mean()