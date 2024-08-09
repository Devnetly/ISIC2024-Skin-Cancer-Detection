import torch
import sys
import os
import warnings
import logging
import pandas as pd
import numpy as np
import h5py
import time
import joblib
sys.path.append('../..')
from argparse import ArgumentParser
from torch import nn,optim
from torch.utils.data import DataLoader,WeightedRandomSampler,ConcatDataset,default_collate,Subset
from dataclasses import dataclass
from tqdm.auto import tqdm
from pandas.errors import DtypeWarning
from definitions import *
from src.utils import seed_everything,score,load
from src.datasets import ISICDataset,ImagesDirectory
from src.models import ResNet,ViT,EfficientNet,MobileNet,Deit
from src.preprocessing import ReinhardAugmentation,RGBToHSV,RGBToLAB
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score
from src.loss import FocalLoss
import albumentations as A
import albumentations.pytorch as AP
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DtypeWarning)
logging.basicConfig(level=logging.INFO)

@dataclass
class Args:
    experiment: str
    epochs: int | None
    num_workers: int | None
    prefetch_factor: int | None

@dataclass
class Config:

    ### Seed
    seed : int = 42, ### Seed for reproducibility

    ### Hyperparameters
    batch_size: int = 32,
    learning_rate : float = 1e-5
    epochs : int = 10
    dropout : float = 0.5
    weight_decay : float = 0.0

    ### Architecture
    model : str = 'resnet18'

    ### Sampling
    pos_weight : float | str = "balanced" ### balanced or float
    num_samples_ratio : float = 1.0 ### Ratio of samples to use
    sampling_type : str = 'dynamic' ### Dynamic ==> new Subset each epoch, Static ==> same Subset each epoch

    ### Dataloaders
    num_workers : int = 4
    prefetch_factor : int = 2

    ### External Data
    external_data : bool = False

    ### Loss
    loss : str = 'bce'
    pos_class_weight : float | None = None ### Weight for positive class
    gamma : float = 2.0
    alpha : float = 0.25

    ### Augmentation & Preprocessing
    img_size : tuple[int,int] = (224,224)
    reinhard_augmentation : bool = True
    random_noise : bool = True
    color_space : str = 'rgb'
    mixup_cutmix : float = 0.0
    color_jitter : bool = False
    others : bool = False

    ### Checkpointing & Early Stopping
    patience : int = 5

def create_transfroms(config : Config) -> tuple[A.Compose, A.Compose]:

    h, w = config.img_size

    color_space = {
        'rgb': RGBToLAB(p=0.0),
        'hsv': RGBToHSV(),
        'lab': RGBToLAB()
    }

    random_noise_args = {
        "min_holes": 1,
        "max_holes": 2,
        "min_height": 0.05,
        "max_height": 0.2,
        "min_width": 0.05,
        "max_width": 0.2,
    }
    
    train_transform = A.Compose([
        color_space[config.color_space],
        ReinhardAugmentation(stats=os.path.join(ISIS_2024_DIR, 'stats.csv'), p=(0.75 * config.reinhard_augmentation)),
        A.OneOf([
            A.Rotate(limit=(90,90),p=1.0),
            A.Rotate(limit=(180,180),p=1.0),
            A.Rotate(limit=(270,270),p=1.0),
        ],p=0.75),
        A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Compose([
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                ],p=1.0)
            ],p=0.75),
        ]),
        A.ColorJitter(p=(0.4 * config.color_jitter)),
        A.RandomResizedCrop(height=h,width=w,scale=(0.8,1.0),p=1.0),
        A.OneOf([
            A.CoarseDropout(p=0.25,fill_value=0,**random_noise_args),
            A.CoarseDropout(p=0.75,fill_value='random',**random_noise_args),
        ],p=(0.75 * config.random_noise)),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.6),
        A.Compose([
            A.Transpose(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2, p=0.75),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        ],p=(0.75 * config.others)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],p=1.0),
        AP.ToTensorV2(p=1.0)
    ])

    val_transform = A.Compose([
        color_space[config.color_space],
        A.Resize(height=h,width=w,p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],p=1.0),
        AP.ToTensorV2(p=1.0)
    ])

    return train_transform, val_transform

def create_datasets(config: Config) :

    train_transform, val_transform = create_transfroms(config)

    hdf5_file = h5py.File(os.path.join(ISIS_2024_DIR, 'images.hdf5'))
    metadata_file = pd.read_csv(os.path.join(ISIS_2024_DIR, 'metadata.csv'))

    splits = joblib.load(os.path.join(DATA_DIR, 'splits.pkl'))

    for fold, (train_idx, val_idx) in enumerate(splits):

        train_data = ISICDataset(
            hdf5_file=hdf5_file,
            metadata_file=metadata_file.iloc[train_idx],
            img_transform=train_transform,
            target_col='target',
            target_transform=float
        )

        if config.external_data:

            external_data = ImagesDirectory(
                root=EXTERNAL_DATA_DIR,
                labels=os.path.join(EXTERNAL_DATA_DIR, 'metadata.csv'),
                img_transform=train_transform,
                target_transform=lambda x : 1.0,
                target_col='benign_malignant',
                image_col='isic_id'
            )

            train_data = ConcatDataset([train_data,external_data])

        val_data = ISICDataset(
            hdf5_file=hdf5_file,
            metadata_file=metadata_file.iloc[val_idx],
            img_transform=val_transform,
            target_col='target',
            target_transform=float
        )

        yield train_data, val_data

class Collate:
    
    def __init__(self,config : Config, training : bool) -> None:

        self.config = config
        self.training = training

        """self.cut_mix_or_mixup = v2.RandomChoice([
            v2.MixUp(num_classes=2),
            v2.CutMix(num_classes=2),
        ], p=[0.5,0.5])"""
    
    def __call__(self,batch : tuple[torch.Tensor,torch.Tensor]) -> tuple[torch.Tensor,torch.Tensor]:

        x,y = default_collate(batch)

        x = x['image']

        if self.training and np.random.rand() < self.config.mixup_cutmix:
            y = y.long()
            x, y = self.cut_mix_or_mixup(x,y)
            y = y[:,1]
        
        return x,y

def create_dataloaders(config: Config):

    train_collate = Collate(config,training=True)
    val_collate = Collate(config,training=False)

    for fold, (train_dataset, val_dataset) in enumerate(create_datasets(config)):

        labels = None

        if config.external_data:
            labels = train_dataset.datasets[0].metadata['target'].to_list() + [1.0] * len(train_dataset.datasets[1])
        else:
            labels = train_dataset.metadata['target'].to_list()

        pos_weight = config.pos_weight

        if pos_weight == 'balanced':
            pos_weight = (len(labels) - sum(labels)) / sum(labels)

        weights = [pos_weight if x == 1 else 1 for x in labels]

        num_samples = int(config.num_samples_ratio * len(labels))

        if config.sampling_type == 'static':

            labels_indices = np.concatenate([
                np.argwhere(np.array(labels) == 1).flatten(),
                np.random.choice(
                    np.argwhere(np.array(labels) == 0).flatten(),
                    num_samples - sum(labels),
                    replace=False
                )
            ])

            train_dataset = Subset(train_dataset, labels_indices)

            weights = [weights[i] for i in labels_indices]

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            sampler=sampler,
            collate_fn=train_collate,
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            collate_fn=val_collate,
        )

        yield train_loader, val_loader

def create_model(config : Config) -> nn.Module:
    
    if config.model.startswith('resnet') or config.model.startswith('resnext'):

        model = ResNet(
            model_name=config.model,
            num_classes=1,
            pretrained=True,
            dropout=config.dropout
        )

    elif 'vit' in config.model:

        model = ViT(
            model_name=config.model,
            num_classes=1,
            pretrained=True,
            dropout=config.dropout
        )

    elif config.model.startswith('efficientnet'):

        model = EfficientNet(
            model_name=config.model,
            num_classes=1,
            pretrained=True,
            dropout=config.dropout
        )
        
    elif config.model.startswith('mobilenet'):
            
        model = MobileNet(
            model_name=config.model,
            num_classes=1,
            pretrained=True,
            dropout=config.dropout
        )

    elif config.model.startswith('deit'):
            
        model = Deit(
            model_name=config.model,
            num_classes=1,
            pretrained=True,
            dropout=config.dropout
        )
    
    else:
        raise ValueError(f"Model {config.model} not supported.")
    
    return model

def create_criteria(config : Config) -> nn.Module:
    if config.loss == 'bce':
        pos_weight = torch.scalar_tensor(config.pos_class_weight) if config.pos_class_weight is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif config.loss == 'focal':
        return FocalLoss(gamma=config.gamma, alpha=config.alpha)
    
def create_optimizer(model: nn.Module, config: Config) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=config.learning_rate,weight_decay=config.weight_decay)

def train(
    config : Config,
    loaders : list[tuple[DataLoader, DataLoader]],
    expirement_dir : str,
    epochs : int,
    device : torch.device
) -> None:

    ### history
    history = []

    for fold, (train_loader, val_loader) in enumerate(loaders):

        fold_dir = os.path.join(expirement_dir, 'checkpoints', f'fold_{fold}')

        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)

        model = create_model(config).to(device)
        criteria = create_criteria(config).to(device)
        optimizer = create_optimizer(model, config)

        print(f"\nTraining Fold {fold} : \n")

        ### Checkpointing
        best_auc = 0.0
        patience = 0

        ### Training Loop
        for epoch in range(epochs):
            
            ### Training Loop
            running_loss = 0.0
            Y = []
            Y_hat = []
            model.train()
            iterator = tqdm(enumerate(train_loader), total=len(train_loader))

            tic = time.time()

            for i, (x, y) in iterator:

                ### Move to device
                x = x.to(device)
                y = y.to(device)

                ### Zero Gradients
                optimizer.zero_grad()

                ### Forward Pass
                y_hat = model(x)

                ### Loss
                loss = criteria(y_hat, y)

                ### Backward Pass
                loss.backward()
                
                ### Update Weights
                optimizer.step()
                y_hat = torch.sigmoid(y_hat)

                ### Metrics
                Y.append(y.detach().to('cpu').numpy())
                Y_hat.append(y_hat.detach().to('cpu').numpy())
                running_loss += loss.detach().to('cpu').item()

                ### Update progress bar
                iterator.set_description(f'Epoch {epoch+1}/{epochs} - Batch Loss: {loss.item()}, Running Loss: {running_loss/(i+1)}')

            toc = time.time()

            ### Evaluation
            Y = np.concatenate(Y)
            Y_hat = np.concatenate(Y_hat)

            pauc = score(Y, Y_hat)
            auc = roc_auc_score(Y, Y_hat)
            f1 = f1_score(Y, Y_hat > 0.5)
            acc = accuracy_score(Y, Y_hat > 0.5)

            df = pd.DataFrame({
                'epoch': [epoch+1],
                'pauc': [pauc],
                'auc': [auc],
                'f1': [f1],
                'acc': [acc],
                'loss': [running_loss/len(train_loader)]
            })

            print()
            print(df)
            print()

            df["split"] = ["train"]
            df["time"] = [toc - tic]
            df["fold"] = [fold]

            history.append(df)

            del Y, Y_hat

            tic = time.time()

            ### Validation Loop
            Y = []
            Y_hat = []
            model.eval()
            iterator = tqdm(enumerate(val_loader), total=len(val_loader))
            running_loss = 0.0
            
            with torch.inference_mode():

                for i, (x, y) in iterator:

                    ### Move to device
                    x = x.to(device)
                    y = y.to(device)

                    ### Forward Pass
                    y_hat = model(x)

                    ### Loss
                    loss = criteria(y_hat, y)
                    y_hat = torch.sigmoid(y_hat)

                    ### Metrics
                    Y.append(y.detach().to('cpu').numpy())
                    Y_hat.append(y_hat.detach().to('cpu').numpy())
                    running_loss += loss.detach().to('cpu').item()

                    ### Update progress bar
                    iterator.set_description(f'Epoch {epoch+1}/{epochs} - Batch Loss: {loss.item()}, Running Loss: {running_loss/(i+1)}')

            toc = time.time()

            ### Evaluation
            Y = np.concatenate(Y)
            Y_hat = np.concatenate(Y_hat)

            pauc = score(Y, Y_hat)
            auc = roc_auc_score(Y, Y_hat)
            f1 = f1_score(Y, Y_hat > 0.5)
            acc = accuracy_score(Y, Y_hat > 0.5)

            df = pd.DataFrame({
                'epoch': [epoch+1],
                'pauc': [pauc],
                'auc': [auc],
                'f1': [f1],
                'acc': [acc],
                'loss': [running_loss/len(val_loader)]
            })

            print()
            print(df)
            print()

            df["split"] = ["val"]
            df["time"] = [toc - tic]
            df["fold"] = [fold]

            history.append(df)

            del Y, Y_hat

            ### Checkpointing
            if pauc > best_auc:

                best_auc = pauc
                patience = 0
                torch.save(model.state_dict(), os.path.join(fold_dir,f'model_fold={fold}.pt'))

                print(f"\n--- Saving model for fold {fold} with pauc {pauc} at epoch {epoch+1} ---\n")

            else:
                patience += 1

            if patience >= config.patience:
                break

    history = pd.concat(history)
    history.to_csv(os.path.join(expirement_dir, 'history.csv'), index=False)

def load_config(expirement : str) -> Config:

    expirement_dir = os.path.join(EXPIREMENTS_DIR, expirement)
    config_path = os.path.join(expirement_dir, 'config.json')

    if not os.path.exists(expirement_dir):
        raise FileNotFoundError(f"Experiment directory {expirement_dir} does not exisv2.")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exisv2.")
    
    return Config(**load(config_path))

def main(args : Args) -> None:

    ### Load config
    expirement_dir = os.path.join(EXPIREMENTS_DIR, args.experiment)
    config = load_config(args.experiment)
    config.epochs = args.epochs if args.epochs is not None else config.epochs

    ### Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Reproducibility
    seed_everything(config.seed)

    ### DataLoading
    loaders = create_dataloaders(config)

    ### Train
    train(
        config=config,
        loaders=loaders,
        expirement_dir=expirement_dir,
        epochs=config.epochs,
        device=device
    )

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--experiment', type=str, required=True)

    ### Override config (Only those that don't effect the results)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--prefetch-factor', type=int, default=2)

    args = parser.parse_args()

    main(args)