import timm
from torch.nn.modules import Module
from .base import BaseModule

class SwinTransformer(BaseModule):

    __models__ = {
        'swin_tiny_patch4_window7_224' : timm.models.swin_transformer.swin_tiny_patch4_window7_224,
        'swin_s3_tiny_224' : timm.models.swin_transformer.swin_s3_tiny_224
    }

    __dims__ = {
        'swin_tiny_patch4_window7_224' : 768,
        'swin_s3_tiny_224' : 768
    }

    def __init__(self, model_name: str, num_classes: int = 1, pretrained: bool = True, dropout: float = 0):
        super().__init__(model_name, num_classes, pretrained, dropout)

    def create_model(self) -> Module:
        model_fn = self.__models__[self.model_name]
        model = model_fn(pretrained=self.pretrained, num_classes=self.num_classes)
        return model
    
    def replace_classifier(self, classifier: Module) -> 'SwinTransformer':
        self.model.head.fc = classifier
        return self
    
    def get_dim(self) -> int:
        return self.__dims__[self.model_name]