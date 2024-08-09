import timm
from torch import nn
from .base import BaseModule

class EfficientNet(BaseModule):

    __models__ = {
        'efficientnet_b0': timm.models.efficientnet.efficientnet_b0,
        'efficientnet_b1': timm.models.efficientnet.efficientnet_b1,
        'efficientnet_b2': timm.models.efficientnet.efficientnet_b2,
        'efficientnet_b3': timm.models.efficientnet.efficientnet_b3,
        'efficientnet_b4': timm.models.efficientnet.efficientnet_b4,
        'efficientnet_b5': timm.models.efficientnet.efficientnet_b5,
        'efficientnet_b6': timm.models.efficientnet.efficientnet_b6,
    }

    __dims__ = {
        'efficientnet_b0': 1280,
        'efficientnet_b1': 1280,
        'efficientnet_b2': 1408,
        'efficientnet_b3': 1536,
        'efficientnet_b4': 1792,
        'efficientnet_b5': 2048,
        'efficientnet_b6': 2304,
        'efficientnet_b7': 2560,
    }

    def __init__(self, model_name: str, num_classes: int = 1, pretrained: bool = True, dropout: float = 0):
        super().__init__(model_name, num_classes, pretrained, dropout)

    def create_model(self):
        model_fn = self.__models__[self.model_name]
        model = model_fn(pretrained=self.pretrained)
        return model
    
    def replace_classifier(self, classifier):
        self.model.classifier = classifier
        return self
    
    def get_dim(self):
        return self.__dims__[self.model_name]