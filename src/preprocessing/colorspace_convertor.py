import cv2
import numpy as np
from PIL import Image

class ColorspaceConvertor:

    def __init__(self, opencv_code: int = cv2.COLOR_RGB2HSV):
        self.opencv_code = opencv_code

    def __call__(self, image: Image.Image) -> Image.Image:

        # Convert image to numpy array
        image = np.array(image)

        # Convert the colorspace
        image = cv2.cvtColor(image, self.opencv_code)

        # Convert image to PIL Image
        image = Image.fromarray(image)

        return image