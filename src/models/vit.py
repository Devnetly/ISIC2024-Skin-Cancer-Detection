import timm
from torch import nn
from .base import BaseModule

class ViT(BaseModule):

    __models__ = {
        'vit_tiny_patch16_224': timm.models.vision_transformer.vit_tiny_patch16_224,
        'vit_small_patch8_224' : timm.models.vision_transformer.vit_small_patch8_224,
        'vit_tiny_r_s16_p8_224' : timm.models.vision_transformer_hybrid.vit_tiny_r_s16_p8_224,
    }

    __dims__ = {
        'vit_tiny_patch16_224' : 192,
        'vit_small_patch8_224' : 384,
        'vit_tiny_r_s16_p8_224' : 192,
    }

    def __init__(
        self,
        model_name: str,
        dropout: float = 0.0,
        num_classes: int = 1,
        pretrained: bool = True
    ):
        
        super(ViT, self).__init__(model_name=model_name, num_classes=num_classes,pretrained=pretrained,dropout=dropout)

    def create_model(self) -> nn.Module:
        model_fn = self.__models__[self.model_name]
        model = model_fn(pretrained=self.pretrained,num_classes=self.num_classes)
        return model
    
    def get_dim(self) -> int:
        return self.__dims__[self.model_name]
    
    def replace_classifier(self, classifier: nn.Module) -> 'ViT':
        self.model.head = classifier
        return self