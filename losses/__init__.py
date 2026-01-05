from losses.laplace_nll_loss import LaplaceNLLLoss
from losses.soft_target_cross_entropy_loss import SoftTargetCrossEntropyLoss

from losses.r1_regularization import R1Regularization
from losses.adv_losses import (
    AdversarialDiscriminatorLoss,
    AdversarialGeneratorLoss,
)
from losses.physics_losses import PhysicsLoss

__all__ = [
    "LaplaceNLLLoss",
    "SoftTargetCrossEntropyLoss",
    "R1Regularization",
    "AdversarialDiscriminatorLoss",
    "AdversarialGeneratorLoss",
    "PhysicsLoss"
]
