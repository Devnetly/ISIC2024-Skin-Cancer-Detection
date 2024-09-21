import timm
from .base import BaseModule
from torch import nn

class NextViT(BaseModule):

    __models__ = {
        'nextvit_small': timm.models.nextvit.nextvit_small,
    }

    __dims__ = {
        'nextvit_small': 1024,
    }

    __weights__ = {
        'nextvit_small' : 'nextvit_small.safetensors'
    }

    def __init__(self, model_name: str, num_classes: int = 1, pretrained: bool = True, dropout: float = 0):
        super().__init__(model_name, num_classes, pretrained, dropout)

    def create_model(self) -> nn.Module:

        model_fn = self.__models__[self.model_name]
        model = model_fn(num_classes=self.num_classes,pretrained=self.pretrained)

        return model
    
    def get_dim(self) -> int:
        return self.__dims__[self.model_name]
    
    def replace_classifier(self, classifier: nn.Module) -> 'NextViT':
        self.model.head.fc = classifier
        return self
