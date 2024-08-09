import torch
from torch import nn,Tensor
from typing import Self

class BaseModule(nn.Module):

    def __init__(self,
        model_name : str,
        num_classes : int = 1,
        pretrained : bool = True,
        dropout : float = 0.0,
    ):

        super(BaseModule, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout = dropout

        self.model = self.create_model()

        if self.dropout > 0.0:

            classifier = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.get_dim(), self.num_classes),
            )
            
        else:
            classifier = nn.Linear(self.get_dim(), self.num_classes)

        self.replace_classifier(classifier)


    def create_model(self) -> nn.Module:
        raise NotImplementedError
    
    def replace_classifier(self, classifier : nn.Module) -> Self:
        raise NotImplementedError
    
    def get_dim(self) -> int:
        raise NotImplementedError
    
    def forward(self, x: Tensor) -> Tensor:
        
        x = self.model(x)

        if self.num_classes == 1:
            x = torch.squeeze(x, dim=-1)

        return x
    
    def predict(self, x: Tensor) -> Tensor:

        x =  self.forward(x)

        if self.num_classes == 1:
            x = torch.sigmoid(x)
        else:
            x = torch.softmax(x, dim=-1)

        return x
    
    def as_backbone(self) -> Self:
        self.replace_classifier(nn.Identity())
        return self