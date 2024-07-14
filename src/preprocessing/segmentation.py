import numpy as np
import cv2
from PIL import Image
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def remove_black_line(image : Image) -> Image.Image:

    arr = np.array(image)
    mask = arr < 25
    mask = np.all(mask, axis=2)
    coords = np.argwhere(~mask)
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()

    return image.crop((y_min, x_min, y_max, x_max))

def khnn_segmentation(image : Image, percentile : float = 10):

    ### Convert the image to numpy array
    x = np.array(image)
    x = x.reshape(-1, 3)
    x = x / 255.0

    ### Train the SOM
    som = MiniSom(64, 64, 3, sigma=1.0, learning_rate=0.5)
    som.train_random(x, num_iteration=1000)

    ### Get the weights
    segmented_image = np.zeros((image.size[1], image.size[0]), dtype=float)
    x = x.reshape(image.size[1], image.size[0], 3)
    for i in range(image.size[1]):
        for j in range(image.size[0]):
            feature_vector = x[i, j, :]
            winner_node = som.winner(feature_vector)
            a,b = winner_node
            weight = som.get_weights()[a, b]
            segmented_image[i, j] = (weight @ feature_vector)

    ### Calculate the  threshold
    threshold = np.percentile(segmented_image, percentile)

    ### Get the binary image
    binary_image = segmented_image < threshold

    ### Preform morphological operations
    kernel = np.ones((5, 5), np.uint8)
    binary_image = cv2.morphologyEx(binary_image.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=3)

    ### Preform flood fill
    h, w = binary_image.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    _,_,binary_image,_ = cv2.floodFill(binary_image, mask, (0, 0), 255)
    binary_image = 1 - binary_image

    return binary_image[:-2, :-2]

def otsu_thresholding(image : Image.Image, seed : tuple = (0, 0)):

    ### Get the mask
    image = np.array(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    ret, image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ### Remove holes with Flood Fill
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    color = (0,0,0)
    _,_,image,_ = cv2.floodFill(image, mask, seed, color)

    ### Invert the image
    image = 1 - image

    ### Remove background noise
    kernel = np.ones((3,3),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=3)

    ### Remove the border
    image = image[:-2,:-2]

    return image

def kmeans_segmentation(image : Image.Image, n_clusters : int = 32, seed : int | None = None):

    ### Convert the image to numpy array
    image = np.array(image)
    shape = image.shape

    ### Reshape the image
    image = image.reshape(-1, 3)

    ### Normalize the image
    scaler = StandardScaler()
    image = scaler.fit_transform(image)

    ### Train the KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    kmeans.fit(image)

    ### Normalize the colors
    colors = scaler.inverse_transform(kmeans.cluster_centers_)

    ### Get the segmented image
    segmented_image = colors[kmeans.labels_].reshape(shape).astype(np.uint8)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    _, segmented_image = cv2.threshold(segmented_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    segmented_image = segmented_image / 255

    ### Apply morphological operations
    kernel = np.ones((3,3),np.uint8)
    segmented_image = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, kernel, iterations=3)

    ### Apply forward fill
    h, w = segmented_image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    segmented_image = segmented_image.astype(np.uint8)
    color = (0,0,0)
    _,_,segmented_image,_ = cv2.floodFill(segmented_image, mask, (0,0), color)
    segmented_image = 1 - segmented_image

    return segmented_image[:-2,:-2]

def variance_matrix(image : Image.Image, ksize : int = 8):

    image = np.array(image)
    
    ### Get the G channel
    image = image[:, :, 1]

    ### Apply Gaussian filter
    image = cv2.GaussianBlur(image, (5, 5), 0)

    ### Calculate the variance matrix
    h, w = image.shape

    variance_matrix = np.zeros((h, w))

    for i in range(h):
        for j in range(w):

            ### Get the window
            x_min = i
            x_max = min(h, i + ksize)
            y_min = j
            y_max = min(w, j + ksize)

            ### Calculate the variance
            variance_matrix[i, j] = np.var(image[x_min:x_max, y_min:y_max])

    ### Calculate the threshold
    threshold = np.std(variance_matrix)

    ### Get the binary image
    binary_image = np.float32(variance_matrix > threshold)

    ### Apply morphological operations
    kernel = np.ones((3,3),np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=3)

    ### Apply forward fill
    h, w = binary_image.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    color = (0,0,0)
    _,_,binary_image,_ = cv2.floodFill(binary_image, mask, (0,0), color)
    binary_image = 1 - binary_image
    binary_image = binary_image[:-2,:-2]

    return binary_image

def segment_image(image : Image.Image, method : str, **kwargs) -> Image.Image:

    image = remove_black_line(image)

    mask = None

    if method == "khnn":
        mask =  khnn_segmentation(image, **kwargs)
    elif method == "otsu":
        mask = otsu_thresholding(image, **kwargs)
    elif method == "kmeans":
        mask = kmeans_segmentation(image, **kwargs)
    elif method == "variance":
        mask = variance_matrix(image, **kwargs)
    else:
        raise ValueError(f"Invalid segmentation method: {method}")
    
    roi = mask[:,:,None] * image

    return Image.fromarray(roi)