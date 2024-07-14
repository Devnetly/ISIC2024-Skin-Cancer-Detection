import json
import random
import torch
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
from typing import Any

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

def score(solution: np.array, submission: np.array, min_tpr : float = 0.8) -> float:
    v_gt = abs(np.asarray(solution)-1)
    
    # flip the submissions to their compliments
    v_pred = -1.0*np.asarray(submission)

    max_fpr = abs(1-min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)
        
    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)
    
    return(partial_auc)