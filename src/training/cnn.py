import sys
import os
import torch
sys.path.append('../..')
import logging
import warnings
import numpy as np
import cv2
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional
from definitions import *
from torch import nn
from torch.utils.data import DataLoader,ConcatDataset,WeightedRandomSampler
from torchvision.transforms import v2 as T
from src.utils import load, seed_everything, load_checkpoint
from src.datasets import ISICDataset,ImagesDirectory
from src.models import ResNet,SimpleCNN
from src.trainer import Trainer
from pandas.errors import DtypeWarning
from torchmetrics.classification import BinaryAccuracy,BinaryF1Score
from src.preprocessing import ReinhardAugmentation,ColorspaceConvertor
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
    seed : int = 42,

    ### Hyperparameters
    batch_size: int = 32,
    learning_rate : float = 1e-5
    weight_decay : float = 1e-6
    epochs : int = 10
    dropout : float = 0.5

    ### Architecture
    model : str = 'resnet18'
    depth : int = 2

    ### Sampling
    minority_class_weight : float = 3.0

    ### Checkpoints
    save_every : Optional[int] = None,
    save_best : Optional[bool] = False

    ### Dataloaders
    num_workers : int = 4
    prefetch_factor : int = 2

    ### External data
    use_external_data : bool = True

    ### Augmentation & Preprocessing
    img_size : tuple[int,int] = (128,128)
    reinhard_augmentation : bool = True
    random_noise : bool = True
    color_space : str = 'rgb'

def create_transforms(config : Config) -> tuple[T.Compose,T.Compose]:
    
    w,h = config.img_size

    train_transforms_list = [
        T.Resize((w,h)),
    ]

    if config.color_space != 'rgb':
        train_transforms_list.append(
            ColorspaceConvertor(
                opencv_code=cv2.COLOR_RGB2HSV if config.color_space == 'hsv' else cv2.COLOR_RGB2LAB
            )
        )

    if config.reinhard_augmentation:
        train_transforms_list.append(
            ReinhardAugmentation(
                stats=os.path.join(ISIS_2024_DIR, 'stats.csv'), 
                p=0.75,
            )
        )

    train_transforms_list.extend([
        T.RandomChoice(transforms=[
            T.RandomRotation(degrees=(0,0)),
            T.RandomRotation(degrees=(90,90)),
            T.RandomRotation(degrees=(180,180)),
            T.RandomRotation(degrees=(270,270)),
        ]),
        T.RandomAffine(degrees=0, translate=(0.3,0.3)),
        T.RandomVerticalFlip(p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.ToImage(), 
    ])

    if config.random_noise:
        train_transforms_list.append(
            T.RandomChoice([
                T.RandomErasing(p=0.5, scale=(0.05, 0.2), ratio=(1.0, 1.0), value='random'),
                T.RandomErasing(p=0.5, scale=(0.05, 0.2), ratio=(1.0, 1.0), value=0)
            ],p=[0.7,0.3])
        )

    train_transforms_list.extend([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_transforms = T.Compose(train_transforms_list)

    val_transforms_list = [
        T.Resize((w,h)),
    ]

    if config.color_space != 'rgb':
        val_transforms_list.append(
            ColorspaceConvertor(
                opencv_code=cv2.COLOR_RGB2HSV if config.color_space == 'hsv' else cv2.COLOR_RGB2LAB
            )
        )

    val_transforms_list.extend([
        T.ToImage(), 
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = T.Compose(val_transforms_list)

    return train_transforms, val_transforms

def create_datasets(config : Config) -> tuple[ConcatDataset,ISICDataset]:

    train_transforms, val_transforms = create_transforms(config)

    isic2024_train_dataset = ISICDataset(
        hdf5_file=os.path.join(ISIS_2024_DIR, 'images.hdf5'),
        metadata_file=os.path.join(ISIS_2024_DIR, 'metadata.csv'),
        return_metadata=False,
        img_transform=train_transforms,
        split='train',
        target_transform=T.Lambda(lambda x: float(x)),
    )

    ### This was not in the first expirement
    if config.use_external_data:

        external_data = ImagesDirectory(
            root=EXTERNAL_DATA_DIR,
            labels=os.path.join(EXTERNAL_DATA_DIR, 'metadata.csv'),
            target_col='benign_malignant',
            image_col='isic_id',
            target_transform=lambda x: 1.0,
            img_transform=train_transforms,
        )

        isic2024_train_dataset = ConcatDataset([isic2024_train_dataset, external_data])

    isic2024_val_dataset = ISICDataset(
        hdf5_file=os.path.join(ISIS_2024_DIR, 'images.hdf5'),
        metadata_file=os.path.join(ISIS_2024_DIR, 'metadata.csv'),
        return_metadata=False,
        img_transform=val_transforms,
        split='val',
        target_transform=T.Lambda(lambda x: float(x)),
    )

    return isic2024_train_dataset, isic2024_val_dataset

def create_dataloaders(config : Config) -> tuple[DataLoader,DataLoader]:

    train_dataset, val_dataset = create_datasets(config)

    if config.use_external_data:
        labels = np.array(train_dataset.datasets[0].get_labels() + [1.0 for _ in range(len(train_dataset.datasets[1]))])
    else:
        labels = np.array(train_dataset.get_labels())

    if config.minority_class_weight == "balanced":
        config.minority_class_weight = (1 - labels).sum() / labels.sum()

    sampler = WeightedRandomSampler(
        weights=[1.0 if label == 0 else config.minority_class_weight for label in labels],
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
    )

    return train_loader, val_loader

def create_network(config : Config) -> nn.Module:
    
    model = ResNet(
        name=config.model,
        dropout_rate=config.dropout,
        depth=config.depth,
        num_classes=1
    )

    return model

def create_criterion(config : Config) -> nn.Module:
    criterion = nn.BCEWithLogitsLoss()
    return criterion

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    ### Create trainer
    logger.info("Creating trainer")

    accuracy = BinaryAccuracy().to(device)
    f1 = BinaryF1Score().to(device)

    trainer = Trainer() \
        .set_checkpoints_folder(expirement_dir) \
        .set_criterion(criterion) \
        .set_optimizer(optimizer) \
        .set_device(device) \
        .set_model(model) \
        .set_save_best(config.save_best) \
        .set_save_every(config.save_every) \
        .add_metric("accuracy",accuracy) \
        .add_metric("f1",f1)
            
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