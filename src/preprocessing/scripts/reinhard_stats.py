import sys
import os
import cv2
import pandas as pd
import numpy as np
sys.path.append('../..')
from src.datasets import ISICDataset
from definitions import *
from tqdm.auto import tqdm
from PIL import Image

def main():
    
    dataset = ISICDataset(
        hdf5_file=os.path.join(ISIS_2024_DIR, 'images.hdf5'),
        metadata_file=os.path.join(ISIS_2024_DIR, 'metadata.csv'),
        split='train',
    )

    df = {
        'mean_l': [],
        'mean_a': [],
        'mean_b': [],
        'std_l': [],
        'std_a': [],
        'std_b': [],
    }

    for image,label in tqdm(dataset):
        
        ### Convert the image to a numpy array
        image = np.array(image)

        ### Convert the image to the LAB colorspace
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        ### Calculate the mean and std of the image
        mean,std = cv2.meanStdDev(image)
        mean = mean.flatten()
        std = std.flatten()

        ### Append the stats to the dataframe
        df['mean_l'].append(mean[0])
        df['mean_a'].append(mean[1])
        df['mean_b'].append(mean[2])
        df['std_l'].append(std[0])
        df['std_a'].append(std[1])
        df['std_b'].append(std[2])

    ### Convert the dictionary to a dataframe
    df = pd.DataFrame(df)

    ### Save the dataframe
    df.to_csv(os.path.join(ISIS_2024_DIR, 'stats.csv'), index=False)

if __name__ == '__main__':
    main()