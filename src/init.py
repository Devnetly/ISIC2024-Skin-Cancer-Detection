import os
import sys
import logging
import warnings
sys.path.append('..')
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from argparse import ArgumentParser
from dataclasses import dataclass
from sklearn.model_selection import train_test_split,GroupKFold
from src.utils import seed_everything
from definitions import *
from pandas.errors import DtypeWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DtypeWarning)
logging.basicConfig(level=logging.INFO)

@dataclass
class Args:
    val_size: float = 0.2
    seed: int = 42

def main(args : Args):

    seed_everything(args.seed)
    
    ### ISIC 2024 ###
    df = pd.read_csv(os.path.join(ISIS_2024_DIR, 'metadata.csv'))

    if 'fold' not in df.columns:

        gkf = GroupKFold(n_splits=6)

        df['fold'] = -1

        for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df['patient_id'])):
            df['fold'].iloc[val_idx] = fold

        df['split'] = df['fold'].apply(lambda x: 'train' if x != 5 else 'val')

        df.to_csv(os.path.join(ISIS_2024_DIR, 'metadata.csv'), index=False)
       
    ### ISIC 2016 ###
    splits_df_path = os.path.join(ISIS_2016_DIR, 'splits.csv')

    if not os.path.exists(splits_df_path):

        files = os.listdir(os.path.join(ISIS_2016_DIR, 'images'))
        files = [f.split('.')[0] for f in files]
        indices = list(range(len(files)))
        _,val_indices = train_test_split(indices, test_size=args.val_size, random_state=args.seed)

        splits_df = pd.DataFrame({
            'image_id': files,
            'split': 'train'
        })

        splits_df['split'].iloc[val_indices] = 'val'

        splits_df.to_csv(splits_df_path, index=False)

    ### ISIC 2019 ###
    df = pd.read_csv(os.path.join(ISIS_2019_DIR, 'labels.csv'))

    if 'split' not in df.columns:
            
        indices = list(range(df.shape[0]))
    
        _,val_indices = train_test_split(indices, test_size=args.val_size, random_state=args.seed)
    
        df['split'] = 'train'
        df['split'].iloc[val_indices] = 'val'
    
        df.to_csv(os.path.join(ISIS_2019_DIR, 'labels.csv'), index=False)
        
if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-size', type=float, default=0.2)

    args = parser.parse_args()

    main(Args(**vars(args)))