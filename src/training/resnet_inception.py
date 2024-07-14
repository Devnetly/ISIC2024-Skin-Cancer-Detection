import sys
import os
import torch
import cv2
sys.path.append('../..')
import logging
import warnings
import pandas as pd
from argparse import ArgumentParser
from dataclasses import dataclass
from definitions import *
from torch import nn
from torch.utils.data import DataLoader,WeightedRandomSampler
from torchvision.transforms import v2 as T
from torchmetrics.classification import BinaryAccuracy,BinaryF1Score,BinaryPrecision,BinaryRecall
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils import load, seed_everything, load_checkpoint
from src.datasets import ISICDataset
from src.models import InceptionResNetV2
from src.trainer import Trainer
from src.augmentation import ReinhardAugmentation
from src.preprocessing import ColorspaceConvertor
from pandas.errors import DtypeWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DtypeWarning)
logging.basicConfig(level=logging.INFO)

@dataclass
class Args:
    experiment: str
    epochs: int | None
    save_every: int | None
    save_best: int | None

@dataclass
class Config:

    ### Seed
    seed : int = 42,  # Seed for reproducibility

    ### Preprocessing
    image_size : tuple[int,int] = (128,128), # The size to resize all the images to.
    columns : list[str] | None = None, # The columns to use from the metadata.
    categories2idx : dict[str, dict[str,int]] = None # The mapping of categories to indexes for each categorical coloumn starting from 0.

    ### Data
    batch_size : int = 32
    num_workers : int = 4
    prefetch_factor : int = 2

    ### Sampling
    mal_sampling_weight : float = 1.0

    ### Architecture
    cat_metadata_config : dict[str, tuple[int,int]] = None
    hidden_dim : int = 256
    heads : int = 16

    ### Hyper-parameters
    metadata_dropout_rate : float = 0.4
    dropout_rate : float = 0.2
    mal_weight : float = 0.5
    learning_rate : float = 1e-3
    weight_decay : float = 1e-4
    epochs : int = 1

    ### Checkpointing
    save_every : int | None = None
    save_best : bool = False

def create_transforms(config : Config) -> tuple[T.Compose,T.Compose,T.Compose]:

    h,w = config.image_size
    categories2idx = config.categories2idx
        
    train_transfroms = T.Compose([
        # ReinhardAugmentation(stats=os.path.join(ISIS_2024_DIR, 'stats.csv'), p=0.5),
        T.RandomChoice(transforms=[
            T.RandomRotation(degrees=(0,0)),
            T.RandomRotation(degrees=(90,90)),
            T.RandomRotation(degrees=(180,180)),
            T.RandomRotation(degrees=(270,270)),
        ]),
        T.RandomVerticalFlip(p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomResizedCrop(size=(h,w), scale=(0.8,1.0)),
        T.Compose([
            T.ToImage(), 
            T.ToDtype(torch.float32, scale=True)
        ]),
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    val_transfroms = T.Compose([
        T.Resize(size=(h, w)),
        T.Compose([
            T.ToImage(), 
            T.ToDtype(torch.float32, scale=True)
        ]),
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    def encode_target(x : pd.Series):

        is_na = x.isna()

        for att,mapper in categories2idx.items():
            x[att] = mapper[x[att]] if not is_na[att] else mapper['<UNK>']
        
        return x

    metadata_transforms = T.Compose([
        T.Lambda(lambda x : x[config.columns] if config.columns is not None else x),
        T.Lambda(encode_target),
        T.Lambda(lambda x : (
            x.drop(categories2idx.keys()).fillna(0.0).to_list(),
            x[categories2idx.keys()].to_dict()
        ))
    ])

    return train_transfroms,val_transfroms,metadata_transforms


def create_datasets(config : Config) -> tuple[ISICDataset,ISICDataset]:

    train_transforms,val_transforms,metadata_transforms = create_transforms(config)

    train_dataset = ISICDataset(
        hdf5_file=os.path.join(ISIS_2024_DIR, 'images.hdf5'),
        metadata_file=os.path.join(ISIS_2024_DIR, 'metadata.csv'),
        return_metadata=True,
        img_transform=train_transforms,
        split='train',
        metadata_transform=metadata_transforms
    )

    val_dataset = ISICDataset(
        hdf5_file=os.path.join(ISIS_2024_DIR, 'images.hdf5'),
        metadata_file=os.path.join(ISIS_2024_DIR, 'metadata.csv'),
        return_metadata=True,
        img_transform=val_transforms,
        split='val',
        metadata_transform=metadata_transforms
    )

    return train_dataset,val_dataset

class Input:

    def __init__(self, data : tuple[torch.Tensor,tuple[torch.Tensor,dict[str,torch.Tensor]]]):
        self.data = data

    def to(self, device : torch.device):
        self.data = (self.data[0].to(device), (self.data[1][0].to(device), { k : v.to(device) for k,v in self.data[1][1].items() }))
        return self
    
    def __getitem__(self, idx : int):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

class Collate:

    def __init__(self, config : Config):
        self.config = config
    
    def __call__(self, batch : list[tuple[tuple[torch.Tensor,tuple[torch.Tensor,dict[str,torch.Tensor]]],torch.Tensor]]):

        images = []
        num_metadatas = []
        cat_metadatas = { k : [] for k in self.config.categories2idx.keys() }
        targets = []
        
        for (image,(num_metadata,cat_metadata)),target in batch:
            
            images.append(image)
            num_metadatas.append(num_metadata)

            for k in cat_metadata.keys():
                cat_metadatas[k].append(cat_metadata[k])

            targets.append(target)

        images = torch.stack(images).float()
        num_metadata = torch.tensor(num_metadatas).float()
        cat_metadata = { k : torch.tensor(v).long() for k,v in cat_metadatas.items() }
        targets = torch.tensor(targets).float()

        return Input((images,(num_metadata,cat_metadata))), targets
    

def create_dataloaders(config : Config) -> tuple[DataLoader,DataLoader]:

    train_dataset, val_dataset = create_datasets(config)
    collate_fn = Collate(config)
    
    sampler = WeightedRandomSampler(
        weights=[1.0 if label == 0 else config.mal_sampling_weight for label in train_dataset.get_labels()],
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


def create_network(config : Config) -> nn.Module:

    model = InceptionResNetV2(
        input_dim=len(config.columns) if config.columns is not None else len(ISICDataset.__cols__),
        cat_metadata_config=config.cat_metadata_config,
        hidden_dim=config.hidden_dim,
        metadata_dropout_rate=config.metadata_dropout_rate,
        dropout_rate=config.dropout_rate,
        heads=config.heads
    )

    return model

def create_criterion(config : Config) -> nn.Module:

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.mal_weight))

    return criterion

def create_optimizer(config : Config,model : nn.Module) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    return optimizer

def create_scheduler(config : Config,optimizer : torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:

    return ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.2, 
        patience=5, 
        verbose=True,
        min_lr=1e-7
    )

def main(args: Args) -> None:

    ### Create logger
    logger = logging.getLogger("main")
    
    ### Experiment directory
    expirement_dir = os.path.join(EXPIREMENTS_DIR, args.experiment)
    config_path = os.path.join(expirement_dir, 'config.json')

    if not os.path.exists(expirement_dir):
        raise FileNotFoundError(f"Experiment directory {expirement_dir} does not exist.")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    ### Load config
    logger.info(f"Loading config from {config_path}")
    config = Config(**load(config_path))

    config.epochs = args.epochs if args.epochs is not None else config.epochs
    config.save_every = args.save_every if args.save_every is not None else config.save_every
    config.save_best = args.save_best if args.save_best is not None else config.save_best

    ### Show config
    logger.info(f"Config: {config}")

    ### Seed everything
    seed_everything(config.seed)

    ### Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    ### Create dataloaders, model, criterion, optimizer
    logger.info("Creating dataloaders, model, criterion, optimizer")
    train_loader, val_loader = create_dataloaders(config)
    model = create_network(config).to(device)
    criterion = create_criterion(config).to(device)
    optimizer = create_optimizer(config,model)
    scheduler = create_scheduler(config,optimizer)

    ### Create trainer
    logger.info("Creating trainer")

    accuracy = BinaryAccuracy().to(device)
    f1 = BinaryF1Score().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)

    trainer = Trainer() \
        .set_checkpoints_folder(expirement_dir) \
        .set_criterion(criterion) \
        .set_optimizer(optimizer) \
        .set_device(device) \
        .set_model(model) \
        .set_save_best(config.save_best) \
        .set_save_every(config.save_every) \
        .add_metric("accuracy",accuracy) \
        .add_metric("f1",f1) \
        .add_metric("precision",precision) \
        .add_metric("recall",recall) \
        .set_scheduler(scheduler,mode="epoch")
            
    ### Load checkpoint
    logger.info(f"Loading checkpoint from {expirement_dir}")
    checkpoint = load_checkpoint(os.path.join(expirement_dir, 'checkpoints'))
    trainer.load(checkpoint)

    ### Start training
    logger.info("Starting training")
    trainer.train(train_loader, val_loader, config.epochs)
    logger.info("Training completed")

    ### Save final checkpoint
    logger.info("Saving final checkpoint")
    trainer.save()

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--experiment', type=str, required=True)

    ### Override config (Only those that don't effect the results)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--save-every', type=int)
    parser.add_argument('--save-best', type=lambda x : x.lower() == 'true', default=False)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--prefetch-factor', type=int, default=2)

    args = parser.parse_args()

    main(args)