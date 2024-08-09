import json
import random
import torch
import numpy as np
import os
from typing import Any
from .evaluation_utils import *

def load(filename : str) -> Any:

    with open(filename, 'r') as file:
        return json.load(file)
    
def seed_everything(seed : int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_checkpoint(checkpoints_dir: str, file : str | None = None) -> dict | None:

    if file is None:
        files = os.listdir(checkpoints_dir)
        files = [f for f in files if f.endswith('.pt')]

        if len(files) == 0:
            return None

        files = sorted(files, key=lambda x: int(x.split(os.extsep)[0].split('_')[1]))
        file = files[-1]
        
    path = os.path.join(checkpoints_dir, file)

    return torch.load(path)