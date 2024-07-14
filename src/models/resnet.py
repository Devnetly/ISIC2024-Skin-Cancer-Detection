import torch
from torch import nn, Tensor
from typing import Optional
from torchvision.models import (
    resnet18, 
    resnet34, 
    resnet50, 
    resnet101,
    resnext50_32x4d,
    resnet152,
    ResNet50_Weights,
    ResNet34_Weights,
    ResNet18_Weights,
    ResNeXt50_32X4D_Weights,
    ResNet101_Weights,
    ResNet152_Weights
)

class ResNet(nn.Module):

    __models__ = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
        'resnext50_32x4d' : resnext50_32x4d
    }

    __weights__ = {
        'resnet18' : ResNet18_Weights.IMAGENET1K_V1,
        'resnet34' : ResNet34_Weights.IMAGENET1K_V1,
        'resnet50' : ResNet50_Weights.IMAGENET1K_V1,
        'resnet101' : ResNet101_Weights.IMAGENET1K_V1,
        'resnext50_32x4d' : ResNeXt50_32X4D_Weights.IMAGENET1K_V1,
        'resnet152' : ResNet152_Weights.IMAGENET1K_V1
    }

    def __init__(self,
        name: str,
        dropout_rate: float = 0.2,
        depth : Optional[int] = None,
        num_classes : int = 1
    ):
        
        super().__init__()

        if name not in ResNet.__models__:
            raise ValueError(f"Unkown model {name}, available models: {list(ResNet.__models__.keys())}.")

        self.name = name
        self.dropout_rate = dropout_rate
        self.depth = depth
        self.num_classes = num_classes

        self.model = ResNet.__models__[name](weights=ResNet.__weights__[name])

        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
            nn.Linear(self.model.fc.in_features, self.num_classes) if self.num_classes > 0 else nn.Identity(),
        )

        self._freeze()

    def _freeze(self) -> None:

        if self.depth is None:
            return

        for param in self.model.parameters():
            param.requires_grad = False

        if self.depth > 0:

            layers = [
                self.model.layer1,
                self.model.layer2,
                self.model.layer3,
                self.model.layer4
            ]

            for layer in layers[-self.depth:]:
                for param in layer.parameters():
                    param.requires_grad = True

            if self.depth == 5:
                for layer in [self.model.conv1,self.model.bn1]:
                    for param in layer.parameters():
                        param.requires_grad = True
                        
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:

        y = self.model(x)

        if self.num_classes == 1:
            y = torch.squeeze(y, dim=-1)

        return y
    
    def predict(self, x: Tensor) -> Tensor:

        x = self.forward(x)

        if self.num_classes == 1:
            x = torch.sigmoid(x)
        else:
            x = torch.softmax(x, dim=-1)

        return x
    
    def as_backbone(self) -> nn.Module:
        self.model.fc = nn.Identity()
        return self