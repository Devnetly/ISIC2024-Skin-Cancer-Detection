from torch.nn.modules import Module
from .base import BaseModule
import timm

class Eva02(BaseModule):

    __models = {
        'eva02_tiny_patch14_224' : timm.models.eva.eva02_tiny_patch14_224,
        'eva02_tiny_patch14_336' : timm.models.eva.eva02_tiny_patch14_336
    }

    __dims__ = {
        'eva02_tiny_patch14_224' : 192,
        'eva02_tiny_patch14_336' : 192
    }

    def __init__(self, model_name: str, num_classes: int = 1, pretrained: bool = True, dropout: float = 0):
        super().__init__(model_name, num_classes, pretrained, dropout)

    def create_model(self) -> Module:
        model_fn = self.__models[self.model_name]
        model = model_fn(pretrained=self.pretrained, num_classes=self.num_classes)
        return model
    
    def get_dim(self) -> int:
        return self.__dims__[self.model_name]
    
    def replace_classifier(self, classifier: Module) -> 'Eva02':
        self.model.head = classifier
        return self