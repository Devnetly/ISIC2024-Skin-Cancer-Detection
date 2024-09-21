import timm
from torch.nn.modules import Module
from .base import BaseModule

class Coat(BaseModule):

    __models__ = {
        'coat_lite_tiny' : (timm.models.coat.coat_lite_tiny,320),
        'coat_lite_mini' : (timm.models.coat.coat_lite_mini,512),
    }

    def __init__(self, model_name: str, num_classes: int = 1, pretrained: bool = True, dropout: float = 0):
        super().__init__(model_name, num_classes, pretrained, dropout)

    def create_model(self) -> Module:
        model_fn, _ = self.__models__[self.model_name]
        model = model_fn(pretrained=self.pretrained)
        return model
    
    def replace_classifier(self, classifier: Module) -> 'Coat':
        self.model.head = classifier
        return self
    
    def get_dim(self) -> int:
        _, dim = self.__models__[self.model_name]
        return dim