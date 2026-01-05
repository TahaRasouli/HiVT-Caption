import torch

class PhysicsLoss:
    @staticmethod
    def compute_jerk_loss(trajs: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jerk (3rd derivative of position).
        High jerk indicates "shaky" or physically unrealistic movement.
        
        Args:
            trajs: Tensor of shape [N, T, 2] (coordinates)
        Returns:
            Mean L2 norm of the jerk vectors.
        """
        if trajs.shape[1] < 4:
            return torch.tensor(0.0, device=trajs.device)

        # Velocity: Δp
        vel = trajs[:, 1:] - trajs[:, :-1]
        
        # Acceleration: Δv
        acc = vel[:, 1:] - vel[:, :-1]
        
        # Jerk: Δa
        jerk = acc[:, 1:] - acc[:, :-1]
        
        # We take the L2 norm of the jerk at each timestep, then average
        jerk_norm = torch.norm(jerk, p=2, dim=-1)
        return jerk_norm.mean()

    @staticmethod
    def compute_curvature_loss(trajs: torch.Tensor) -> torch.Tensor:
        """
        Optional: Measures how sharp the turns are. 
        Useful if you find the GAN making 'zigzag' motions.
        """
        vel = trajs[:, 1:] - trajs[:, :-1]
        # Cross product of consecutive velocity vectors (2D)
        # approximates the local curvature
        return torch.tensor(0.0, device=trajs.device) # Placeholder for future use
