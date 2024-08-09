import cv2
import numpy as np
from PIL import Image
from functools import partial
from albumentations.core.transforms_interface import ImageOnlyTransform

class ColorspaceConvertor(ImageOnlyTransform):

    def __init__(self, opencv_code: int = cv2.COLOR_RGB2HSV,p=1.0):

        super(ColorspaceConvertor, self).__init__(p=p)

        self.opencv_code = opencv_code


    def apply(self, img : np.array, **params) -> Image.Image:
        
        # Convert the colorspace
        image = cv2.cvtColor(img, self.opencv_code)

        return image

RGBToHSV = partial(ColorspaceConvertor, opencv_code=cv2.COLOR_RGB2HSV)
RGBToLAB = partial(ColorspaceConvertor, opencv_code=cv2.COLOR_RGB2LAB)