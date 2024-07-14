from typing import Any, Callable, Optional
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import h5py
import io
import warnings
from pandas.errors import DtypeWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DtypeWarning)

__cols__ = ['age_approx','sex','anatom_site_general','clin_size_long_diam_mm','tbp_lv_A']
__cols__ += ['tbp_lv_Aext','tbp_lv_B','tbp_lv_Bext','tbp_lv_C','tbp_lv_Cext','tbp_lv_H','tbp_lv_Hext','tbp_lv_L']
__cols__ += ['tbp_lv_Lext','tbp_lv_areaMM2','tbp_lv_area_perim_ratio','tbp_lv_color_std_mean','tbp_lv_deltaA']
__cols__ += ['tbp_lv_deltaB','tbp_lv_deltaL','tbp_lv_deltaLB','tbp_lv_deltaLBnorm','tbp_lv_eccentricity','tbp_lv_location']
__cols__ += ['tbp_lv_minorAxisMM','tbp_lv_nevi_confidence','tbp_lv_norm_border','tbp_lv_norm_color']
__cols__ += ['tbp_lv_perimeterMM','tbp_lv_radial_color_std_max','tbp_lv_stdL','tbp_lv_stdLExt','tbp_lv_symm_2axis']
__cols__ += ['tbp_lv_symm_2axis_angle','tbp_lv_x','tbp_lv_y','tbp_lv_z']

class ISICDataset(Dataset):

    __cols__ = __cols__
    
    def __init__(self, 
        hdf5_file : str | h5py.File,
        metadata_file : str | pd.DataFrame,
        img_transform : Optional[Callable] = None,
        metadata_transform : Optional[Callable] = None,
        target_transform : Optional[Callable] = None,
        return_metadata : bool = False,
        split : Optional[str] = None,
        fold : Optional[int] = None,
        mode : str = "train"
    ) -> None:
        
        super().__init__()
        
        self.hdf5_file = hdf5_file
        self.metadata_file = metadata_file
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.metadata_transform = metadata_transform
        self.mode = mode
        self.fold = fold
        
        self.metadata = pd.read_csv(self.metadata_file) if isinstance(self.metadata_file, str) else self.metadata_file
        self.return_metadata = return_metadata

        if split is not None:
            if fold is None:
                self.metadata = self.metadata[self.metadata['split'] == split]
            else:
                if split == "train":
                    self.metadata = self.metadata[(self.metadata['split'] == "train") & (self.metadata['fold'] != fold)]
                else:
                    self.metadata = self.metadata[(self.metadata['split'] == "train") & (self.metadata['fold'] == fold)]
        
        self.hdf5 = h5py.File(self.hdf5_file, "r") if isinstance(self.hdf5_file, str) else self.hdf5_file

    def get_labels(self) -> list[int]:
        return self.metadata['target'].tolist()
       
    def __len__(self):
        return self.metadata.shape[0]
    
    def __getitem__(self, index : int) -> tuple[tuple[Any,Any],Any]:
        
        ### Get the metadata row
        row = self.metadata.iloc[index]
        
        ### Get the target
        target = row['target'] if self.mode != "test" else 0.0
        
        ### The image
        image_name = row['isic_id']
        dataset = self.hdf5[image_name]
        buffer = dataset[()]
        image_file = io.BytesIO(buffer)
        img = Image.open(image_file)

        ### The metadata
        metadata = row[ISICDataset.__cols__]
        
        ### Apply the transformations
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.img_transform is not None:
            img = self.img_transform(img)
            
        if self.metadata_transform is not None and self.return_metadata:
            metadata = self.metadata_transform(metadata)
        
        if self.return_metadata:
            return (img,metadata), target
        
        return img, target