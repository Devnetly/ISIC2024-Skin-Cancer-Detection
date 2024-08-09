import timm
from torch import nn
from .base import BaseModule

class Deit(BaseModule):

    __models__ = {
        'deit_tiny_patch16_224' : timm.models.deit.deit_tiny_patch16_224,
        'deit3_small_patch16_224.fb_in22k_ft_in1k' : timm.models.deit.deit3_small_patch16_224,
        'deit3_small_patch16_384.fb_in22k_ft_in1k' : timm.models.deit.deit3_small_patch16_384,
    }

    __dims__ = {
        'deit_tiny_patch16_224' : 192,
        'deit3_small_patch16_224.fb_in22k_ft_in1k' : 384,
        'deit3_small_patch16_384.fb_in22k_ft_in1k' : 384,
    }

    def __init__(self, model_name: str, num_classes: int = 1, pretrained: bool = True, dropout: float = 0):
        super().__init__(model_name, num_classes, pretrained, dropout)

    def create_model(self) -> nn.Module: 
        model_fn = self.__models__[self.model_name]
        model = model_fn(pretrained=self.pretrained)
        return model
    
    def replace_classifier(self, classifier: nn.Module) -> 'Deit':
        self.model.head = classifier
        return self
    
    def get_dim(self) -> int:
        return self.__dims__[self.model_name]