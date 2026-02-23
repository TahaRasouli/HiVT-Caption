import torch

class PhysicsLoss:
    @staticmethod
    def compute_jerk_loss(trajs: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jerk (3rd derivative of position).
        High jerk indicates "shaky" or physically unrealistic movement.
        
        Args:
            trajs: Tensor of shape [N, T, 2] (valid agent coordinates)
        Returns:
            Mean L2 norm of the jerk vectors, safe-guarded against NaN gradients.
        """
        # If the trajectory is too short to compute a 3rd derivative, return 0
        if trajs.shape[1] < 4:
            return torch.tensor(0.0, device=trajs.device, requires_grad=True)

        # 1. Velocity: Change in position (Δp) -> Shape: [N, T-1, 2]
        vel = trajs[:, 1:] - trajs[:, :-1]
        
        # 2. Acceleration: Change in velocity (Δv) -> Shape: [N, T-2, 2]
        acc = vel[:, 1:] - vel[:, :-1]
        
        # 3. Jerk: Change in acceleration (Δa) -> Shape: [N, T-3, 2]
        jerk = acc[:, 1:] - acc[:, :-1]
        
        # CRITICAL FIX: Safe L2 norm to prevent NaN gradients.
        # Adding 1e-6 inside the square root prevents the derivative (1 / 2*sqrt(x)) 
        # from dividing by zero when the jerk is exactly [0.0, 0.0].
        jerk_norm = torch.sqrt(jerk.pow(2).sum(dim=-1) + 1e-6)
        
        return jerk_norm.mean()

    @staticmethod
    def compute_curvature_loss(trajs: torch.Tensor) -> torch.Tensor:
        """
        Optional: Measures how sharp the turns are. 
        Useful if you find the GAN making 'zigzag' motions.
        """
        # Placeholder for future use. 
        # Note: If implemented, ensure you use the exact same `+ 1e-6` safe-norm trick!
        return torch.tensor(0.0, device=trajs.device, requires_grad=True)