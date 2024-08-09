import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import roc_curve, auc,accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
from torch.utils.data import DataLoader
from torch import nn

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

def calc_metrics(ytrue : np.ndarray, yhat : np.ndarray):

    yhard = np.int32(yhat > 0.5)

    acc = accuracy_score(ytrue, yhard)
    f1 = f1_score(ytrue, yhard, zero_division=0)
    prec = precision_score(ytrue, yhard, zero_division=0)
    rec = recall_score(ytrue, yhard, zero_division=0)
    pauc = score(ytrue, yhat)

    return pd.Series({'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec, 'partial_auc': pauc})

def plot_confusion_matrix(ytrue : np.ndarray, yhat : np.ndarray) -> None:
    cm = confusion_matrix(ytrue, np.int32(yhat > 0.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    

def predict(
    model: nn.Module, 
    dataloader: DataLoader,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) ->tuple[np.ndarray,np.ndarray]:

    model.eval()

    Y_true = []
    Y_hat = []

    with torch.inference_mode():

        for x, y in dataloader:

            x = x.to(device)
            y_hat = model(x).detach().cpu()

            Y_hat.append(y_hat.numpy())
            Y_true.append(y.numpy())

    Y_true = np.concatenate(Y_true)
    Y_hat = np.concatenate(Y_hat)

    return Y_true, Y_hat

def predict_all(
    models: list[nn.Module],
    loaders: list[DataLoader] | DataLoader,
) -> list[pd.DataFrame] | pd.DataFrame:

    if isinstance(loaders, DataLoader):
        ### Return a dataframe
        df = {}

        for i, model in enumerate(models):
            Y_true, Y_hat = predict(model, loaders)
            df[f'model_{i}'] = pd.Series(Y_hat)

        df['Y_true'] = pd.Series(Y_true)

        return pd.DataFrame(df)
    
    if len(models) != len(loaders):
        raise ValueError('Models and loaders must have the same length')
    
    ### Return a list of dataframes

    dfs = []

    for i, (model, loader) in enumerate(zip(models, loaders)):
        Y_true, Y_hat = predict(model, loader)
        df = pd.DataFrame({'Y_true': Y_true, f'model_{i}': Y_hat})
        dfs.append(df)

    return dfs

def evaluate(
    preds : pd.DataFrame | list[pd.DataFrame],
) -> pd.DataFrame | pd.Series:
    
    if isinstance(preds, pd.DataFrame):
        return calc_metrics(preds['Y_true'], preds['target'])
    
    results_df = pd.DataFrame()
    
    for i,df in enumerate(preds):
        results_df[f'model_{i}'] = calc_metrics(df['Y_true'], df['target'])

    results_df['mean'] = results_df.mean(axis=1)
    results_df['std'] = results_df.std(axis=1)

    return results_df