import pandas as pd
import numpy as np
import cv2
import random
from PIL import Image
from typing import Callable
from albumentations.core.transforms_interface import ImageOnlyTransform

class ReinhardAugmentation(ImageOnlyTransform):

    def __init__(self, 
        stats : str,
        p: float = 0.5,
    ):
        
        super(ReinhardAugmentation, self).__init__(p=p)

        self.stats = pd.read_csv(stats)
        self.p = p

    def apply(self, img : np.ndarray, **paramse) -> Image.Image:

        ### Convert the image to the LAB colorspace
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        ### Sample a random mean and std from the stats
        idx = random.randint(0, len(self.stats)-1)
        mean_r, mean_g, mean_b, std_r, std_g, std_b = self.stats.iloc[idx].values

        ### Convert to tensors
        mean = np.array([mean_r, mean_g, mean_b]).reshape(1,1,3)
        std = np.array([std_r, std_g, std_b]).reshape(1,1,3)

        ### Calculate the mean and std of the image
        img_mean,img_std = cv2.meanStdDev(img)
        img_mean = img_mean.reshape(1,1,3)
        img_std = img_std.reshape(1,1,3)

        ### Preform color transfer
        img = (img - img_mean) / img_std
        img = img * std + mean

        ### Clip the image
        img = np.where(img > 255, 255, img)
        img = np.where(img < 0, 0, img)

        ### Convert the image back to the RGB colorspace
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        ### Return the image
        return img