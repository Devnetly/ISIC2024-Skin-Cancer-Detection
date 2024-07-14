import torch
import timm
from torch import nn,Tensor

class SoftAttention(nn.Module):

    def __init__(self,
        channels : int,
        heads : int,
    ) -> None:
        
        super().__init__()

        self.channels = channels
        self.heads = heads

        self.conv3d = nn.Conv3d(in_channels=1, out_channels=self.heads, kernel_size=(channels, 3, 3), padding=(0, 1, 1))
        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x : Tensor):
        
        # x : (B,C,H,W)
        B,C,H,W = x.size()
        og_x = x

        x = x.unsqueeze(1) # x : (B,1,C,H,W)

        x = self.conv3d(x) # x : (B,heads,1,H,W)
        x = self.relu(x) # x : (B,heads,1,H,W)

        x = x.squeeze(2) # x : (B,heads,H,W)
        x = x.reshape(x.size(0),self.heads,-1) # x : (B,heads,H*W)

        x = self.softmax(x) # x : (B,heads,H*W)
        alpha = x.reshape(x.size(0), self.heads, H, W) # alpha : (B,heads,H,W)

        x = alpha.sum(dim=1, keepdim=True) # x : (B,1,H,W)

        u = og_x * x # u : (B,C,H,W)

        return u, alpha

class InceptionResNetV2(nn.Module):

    def __init__(self,
        heads : int = 16,
        dropout_rate : float = 0.2,
        num_classes : int = 1
    ) -> None:

        super().__init__()

        model = timm.create_model('inception_resnet_v2',pretrained=True)

        self.cnn = nn.Sequential(
            model.conv2d_1a,
            model.conv2d_2a,
            model.conv2d_2b,
            model.maxpool_3a,
            model.conv2d_3b,
            model.conv2d_4a,
            model.maxpool_5a,
            model.mixed_5b,
            model.repeat,
            model.mixed_6a,
            model.repeat_1,
            model.mixed_7a,
            model.repeat_2,
            model.block8.branch0,
        )

        self.attention = SoftAttention(channels=192,heads=heads)

        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(6144,num_classes)
        )

    def forward(self,x : Tensor) -> Tensor:
        

        ### 1- Process the image
        cnn_out = self.cnn(x)
        features, att_map = self.attention(cnn_out)

        ### 2- Apply max pooling
        x1 = self.max_pool1(cnn_out)
        x2 = self.max_pool2(features)

        ### 3- Concatenate the features
        x = torch.cat([x1, x2], dim=1)

        ### 4- Process the concatenated features
        x = self.relu(x)
        x = self.flatten(x)

        ### 5- Make prediction
        x = self.classifier(x)

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
        self.classifier = nn.Identity()
        return self