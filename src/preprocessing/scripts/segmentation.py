import sys
import os
sys.path.append('../..')
from argparse import ArgumentParser
from definitions import *
from dataclasses import dataclass
from src.utils import seed_everything
from src.datasets import ISICDataset
from dataclasses import dataclass
from segmentation import segment_image
from tqdm.auto import tqdm
from multiprocessing import Pool
from PIL import Image

@dataclass
class Args:
    seed: int
    method: str
    output: str

def main(args : Args):
    
    seed_everything(args.seed)

    dataset = ISICDataset(
        hdf5_file=os.path.join(ISIS_2024_DIR, "images.hdf5"),
        metadata_file=os.path.join(ISIS_2024_DIR, "metadata.csv"),
    )
    
    for i, (image,_) in tqdm(enumerate(dataset), total=len(dataset)):

        segmented_image = segment_image(image, args.method)
        filename = dataset.metadata.iloc[i].isic_id
        segmented_image.save(os.path.join(DATA_DIR,args.output, f"{filename}.png"))

    
if __name__ == '__main__':
    
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="kmeans")
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    main(Args(**vars(args)))    