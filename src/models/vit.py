import torch
import timm
from torch import nn,Tensor

class ViT(nn.Module):

    __models__ = timm.list_models(filter='vit*')

    def __init__(
        self,
        name: str,
        num_classes: int = 1,
    ):
        
        super().__init__()
        
        self.num_classes = num_classes
        self.name = name

        self.model = timm.create_model(
            model_name=name,
            num_classes=1,
            in_chans=3,
            pretrained=True
        )

    def forward(self, x: Tensor) -> Tensor:
        
        x = self.model(x)

        if self.num_classes == 1:
            x = torch.squeeze(x, dim=-1)

        return x
    
    def predict(self, x: Tensor) -> Tensor:
        
        x = self.forward(x)
        
        if self.num_classes == 1:
            x = torch.sigmoid(x)
        else:
            x = torch.softmax(x, dim=-1)
        
        return x
    
    def as_backbone(self) -> nn.Module:
        self.model.head = nn.Identity()
        return self