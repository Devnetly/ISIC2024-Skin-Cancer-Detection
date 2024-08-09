import torch
from torch import nn,Tensor 
from .metadata_net import MetadataNet

class MetadataCNN(nn.Module):
    
    def __init__(self,
        metadata_net : MetadataNet,
        backbone : nn.Module,
        num_classes : int = 1
    ) -> None:
        
        super().__init__()

        self.num_classes = num_classes
        self.backbone_dim = backbone.get_dim()

        self.metadata_net = metadata_net
        self.backbone = backbone   

        self.fc = nn.Linear(
            in_features=backbone.get_dim() + metadata_net.hidden_dim,
            out_features=num_classes
        )

    def forward(self,x : tuple[Tensor, tuple[Tensor,dict[str,Tensor]]]) -> Tensor:

        ### Unpack the input
        x, metadata = x

        ### Process the metadata
        metadata = self.metadata_net(metadata)

        ### Process the image
        x = self.backbone(x)

        ### Concatenate the image and metadata
        x = torch.cat([x, metadata], dim=-1)

        ### Process the concatenated features
        x = self.fc(x)

        if self.num_classes == 1:
            x = torch.squeeze(x, dim=-1)

        return x
    
    def predict(self,x : tuple[Tensor, tuple[Tensor,dict[str,Tensor]]]) -> Tensor:

        x = self.forward(x)

        if self.num_classes == 1:
            x = torch.sigmoid(x)
        else:
            x = torch.softmax(x, dim=-1)

        return x
    
    def as_backbone(self) -> nn.Module:
        self.fc = nn.Identity()
        return self
    
    