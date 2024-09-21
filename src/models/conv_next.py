import timm
from torch import nn
from .base import BaseModule

class ConvNext(BaseModule):

    __models__ = {
        'convnext_tiny': timm.models.convnext.convnext_tiny,
        'convnext_small': timm.models.convnext.convnext_small,
        'convnext_base': timm.models.convnext.convnext_base,
        'convnext_large': timm.models.convnext.convnext_large,
    }

    __dims__ = {
        'convnext_tiny': 768,
        'convnext_small': 768,
        'convnext_base': 1024,
        'convnext_large': 1536,
    }

    def __init__(self, model_name: str, num_classes: int = 1, pretrained: bool = True, dropout: float = 0):
        super().__init__(model_name, num_classes, pretrained, dropout)

    def create_model(self) -> nn.Module:
        model_fn = self.__models__[self.model_name]
        model = model_fn(pretrained=self.pretrained)
        return model
    
    def replace_classifier(self, classifier: nn.Module) -> 'ConvNext':
        self.model.head.fc = classifier
        return self
    
    def get_dim(self) -> int:
        return self.__dims__[self.model_name]