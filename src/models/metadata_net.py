import torch
from torch import nn,Tensor
from typing import Self

class CatMetadataNet(nn.Module):
    """
    A simple network that embeds categorical metadata into a single tensor.
    """
    
    def __init__(self, config : dict[str,tuple[int,int]]):
        """
        Args:
            config : dict[str,tuple[int,int]] : A dictionary mapping metadata keys to a tuple of (num_embeddings,embedding_dim).
        """

        super().__init__()

        self.input_dim = len(config)
        self.output_dim = sum([v[1] for v in config.values()])

        self.net = nn.ModuleDict()

        for key,(num_embeddings,embedding_dim) in config.items():
            self.net[key] = nn.Embedding(num_embeddings,embedding_dim)
                
    def forward(self, x : dict[str,Tensor]) -> Tensor:
        """
        Args:
            x : tuple[Tensor,Tensor] : A tuple containing the metadata tensor and the isna tensor.
        """
                        
        y = []

        for key, embedding in self.net.items():
            y.append(embedding(x[key]))

        y = torch.cat(y,dim=-1)

        return y
    
class MetadataBlock(nn.Module):

    def __init__(self,
        input_dim : int,
        output_dim : int,
    ) -> None:
        
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim,output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x : Tensor) -> Tensor:

        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
    
class MetadataNet(nn.Module):
    """
    A simple network that processes numerical and categorical metadata.
    """

    __models__ = {
        'matadata_net'
    }

    def __init__(self,
        input_dim : int,
        cat_metadata_config : dict[str,tuple[int,int]],
        hidden_dim : int = 192,
        num_layers : int = 3,
        dropout_rate : float = 0.2,
        num_classes : int = 1
    ):
        """
        Args:
            input_dim : int : The total number of metadata columns.
            cat_metadata_config : dict[str,tuple[int,int]] : A dictionary mapping metadata keys to a tuple of (num_embeddings,embedding_dim).
            hidden_dim : int : The hidden dimension of the network.
            dropout_rate : float : The dropout rate of the network.
        """
        
        super().__init__()

        self.input_dim = input_dim
        self.num_cols_count = input_dim - len(cat_metadata_config)
        self.hidden_dim = hidden_dim
        self.cat_metadata_config = cat_metadata_config
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        self.cat_net = CatMetadataNet(cat_metadata_config)
        
        self.num_net = nn.Sequential(*[
            MetadataBlock(
                input_dim=(self.num_cols_count + self.cat_net.output_dim if i == 0 else hidden_dim),
                output_dim=hidden_dim,
            ) for i in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dim,num_classes)
        
    def forward(self, x : tuple[Tensor,dict[str,Tensor]]) -> Tensor:
        
        ### Unpack the input
        num,cat = x

        ### Embed the categorical metadata
        cat_tensor = self.cat_net(cat)

        ### Concatenate the numerical and categorical metadata
        tensor = torch.cat([num, cat_tensor],dim=-1)
        
        ### Process the metadata
        tensor = self.num_net(tensor)

        ### Make the final prediction
        tensor = self.dropout(tensor)
        tensor = self.fc(tensor)

        if self.num_classes == 1:
            tensor = tensor.squeeze(-1)

        return tensor
    
    def predict(self, x : tuple[Tensor,dict[str,Tensor]]) -> Tensor:
        """
        Args:
            x : tuple[Tensor,dict[str,Tensor]] : A tuple containing the numerical tensor and the metadata dictionary.
        Returns:
            Tensor : The model predictions.
        """
        
        y = self.forward(x)

        if self.num_classes == 1:
            y = torch.sigmoid(y)
        else:
            y = torch.softmax(y,dim=-1)
        
        return y
    
    def as_backbone(self) -> Self:
        """
        Returns:
            nn.Module : The backbone of the model.
        """
        
        self.fc = nn.Identity()
        return self