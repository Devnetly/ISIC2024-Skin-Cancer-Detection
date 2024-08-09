from torch import nn
from .base import BaseModule
from torchvision.models import (
    resnet18, 
    resnet34, 
    resnet50, 
    resnet101,
    resnext50_32x4d,
    ResNet50_Weights,
    ResNet34_Weights,
    ResNet18_Weights,
    ResNeXt50_32X4D_Weights,
    ResNet101_Weights,
)

class ResNet(BaseModule):

    __models__ = {
        'resnet18': (resnet18, ResNet18_Weights.IMAGENET1K_V1, 512),
        'resnet34': (resnet34, ResNet34_Weights.IMAGENET1K_V1, 512),
        'resnet50': (resnet50, ResNet50_Weights.IMAGENET1K_V1, 2048),
        'resnet101': (resnet101, ResNet101_Weights.IMAGENET1K_V1, 2048),
        'resnext50_32x4d': (resnext50_32x4d, ResNeXt50_32X4D_Weights.IMAGENET1K_V1, 2048),
    }

    def __init__(self, model_name: str, num_classes: int = 1, pretrained: bool = True, dropout: float = 0):
        super().__init__(model_name, num_classes, pretrained, dropout)

    def create_model(self) -> nn.Module:

        model_fn, weights, _ = self.__models__[self.model_name]

        model = None

        if self.pretrained:
            model = model_fn(weights=weights)
        else:
            model = model_fn()

        return model
    
    def replace_classifier(self, classifier: nn.Module) -> 'ResNet':
        self.model.fc = classifier
        return self
    
    def get_dim(self) -> int:
        return self.__models__[self.model_name][2]