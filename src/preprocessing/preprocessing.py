import cv2
import numpy as np
from PIL import Image

def preprocess_image(image : Image.Image):

    # Convert image to numpy array
    image = np.array(image)

    # Remove hair from the image
    unhaired_image = hair_remove(image)

    # Apply unsharp filter
    unsharp_image = unsharp_filter(unhaired_image, image)

    # Convert image to PIL Image
    unsharp_image = Image.fromarray(unsharp_image)

    return unsharp_image

def hair_remove(image):
    # convert image to grayScale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # kernel for morphologyEx
    kernel = np.ones((10, 10), np.uint8)

    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # apply thresholding to blackhat
    _,threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # inpaint with original image and threshold image
    unhaired_image = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)

    return unhaired_image

def unsharp_filter(image, original, sigma=10.0, alpha=1.5, beta=-0.5, gamma=0):
    # Apply Gaussian Blur
    gaussian = cv2.GaussianBlur(image, (image.shape[0], image.shape[1]), sigma)

    # Calculate the Unsharp Mask
    unsharp_image = cv2.addWeighted(image, alpha, gaussian, beta, gamma, original)

    return unsharp_image
        