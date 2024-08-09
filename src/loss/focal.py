from torchvision.ops import sigmoid_focal_loss
from torch import Tensor
from torch.nn import Module

class FocalLoss(Module):

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction :str = 'mean') -> None:

        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma, reduction='mean')