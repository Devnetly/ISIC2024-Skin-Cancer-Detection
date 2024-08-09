import timm
from .base import BaseModule



class MobileNet(BaseModule):

    __models__ = {
        'mobilenetv3_large_075' : timm.models.mobilenetv3.mobilenetv3_large_075,
        'mobilenetv3_large_100' : timm.models.mobilenetv3.mobilenetv3_large_100,
        'mobilenetv3_small_050' : timm.models.mobilenetv3.mobilenetv3_small_050,
        'mobilenetv3_small_075' : timm.models.mobilenetv3.mobilenetv3_small_075,
        'mobilenetv3_small_100' : timm.models.mobilenetv3.mobilenetv3_small_100,
    }

    __dims__ = {
        'mobilenetv3_large_075' : 1280,
        'mobilenetv3_large_100' : 1280,
        'mobilenetv3_small_050' : 1024,
        'mobilenetv3_small_075' : 1024,
        'mobilenetv3_small_100' : 1024,
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
        return 1024
