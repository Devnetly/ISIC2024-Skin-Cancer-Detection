import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Optional,Callable

class ImagesDirectory(Dataset):

    def __init__(self,
        root : str,
        labels : str,
        img_transform : Optional[Callable] = None,
        target_transform : Optional[Callable] = None,
        target_col : str = 'taget',
        image_col : str = 'image',
        split : str | None = None
    ) -> None:
        
        super().__init__()

        self.root = root
        self.labels = labels
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.split = split
        self.target_col = target_col
        self.image_col = image_col

        self.labels_df = pd.read_csv(self.labels)

        if split is not None:
            self.labels_df = self.labels_df[self.labels_df['split'] == split]

    def __len__(self) -> int:
        return len(self.labels_df)
    
    def __getitem__(self, idx: int) -> dict:
            
        row = self.labels_df.iloc[idx]
        name = row[self.image_col]
        img_name = name + '.jpg'
    
        image_path = os.path.join(self.root, img_name)
        label = row[self.target_col]
    
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        if self.img_transform is not None:
            image = self.img_transform(image=image)

        if self.target_transform is not None:
            label = self.target_transform(label)
    
        return image, label