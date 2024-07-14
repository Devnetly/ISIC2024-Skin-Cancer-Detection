import torch
import time
import os
from tqdm.auto import tqdm
from torch import nn,Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import Metric
from typing import Optional,Any,Self
from .batch_results import BatchResults
from .history import History

class Trainer:

    def __init__(self, 
        model: Optional[nn.Module] = None, 
        criterion: Optional[nn.Module] = None, 
        optimizer: Optional[Optimizer] = None, 
        scheduler: Optional[_LRScheduler] = None, 
        scheduler_mode: Optional[str] = "epoch",
        metrics: Optional[dict[str, Metric]] = None,
        device: Optional[torch.device] = None,
        checkpoints_folder : Optional[str] = None,
        start_epoch: Optional[int] = 0,
        save_best: Optional[bool] = False,
        save_every: Optional[int] = None
    ) -> None:
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_mode = scheduler_mode
        self.metrics = metrics if metrics is not None else {}
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_epoch = start_epoch
        self.checkpoints_folder = None
        self.history = None
        self.save_best = save_best
        self.save_every = save_every
        self.epochs = 0

        if checkpoints_folder is not None:
            self.set_checkpoints_folder(checkpoints_folder)


    def set_model(self, model: nn.Module) -> Self:
        self.model = model
        return self
    
    def set_criterion(self, criterion: nn.Module) -> Self:
        self.criterion = criterion
        return self
    
    def set_optimizer(self, optimizer: Optimizer) -> Self:
        self.optimizer = optimizer
        return self
    
    def set_scheduler(self, scheduler: _LRScheduler, mode : str = "epoch") -> Self:
        self.scheduler = scheduler
        self.scheduler_mode = mode
        return self
    
    def add_metric(self, name : str, metric: Metric) -> Self:

        self.metrics[name] = metric

        self.history = History(
            header=["epoch", "time", "loss"] + list(self.metrics.keys()), 
            filename=os.path.join(self.checkpoints_folder, "history.csv"),
            eager=True
        )

        return self
    
    def set_device(self, device: torch.device) -> Self:
        self.device = device
        return self
    
    def set_checkpoints_folder(self, checkpoints_folder: str) -> Self:

        self.checkpoints_folder = checkpoints_folder

        self.history = History(
            header=["epoch", "time", "loss"] + list(self.metrics.keys()), 
            filename=os.path.join(checkpoints_folder, "history.csv"),
            eager=True
        )

        return self
    
    def set_start_epoch(self, start_epoch: int) -> Self:
        self.start_epoch = start_epoch
        return self
    
    def set_save_best(self, save_best: bool) -> Self:
        self.save_best = save_best
        return self
    
    def set_save_every(self, save_every: int) -> Self:
        self.save_every = save_every
        return self
    
    def _check(self) -> None:
        assert self.model is not None, "Model is not set"
        assert self.criterion is not None, "Criterion is not set"
        assert self.optimizer is not None, "Optimizer is not set"
        assert self.checkpoints_folder is not None, "Checkpoints folder is not set"

    def _format_dict(self, results : dict[str, Any]) -> str:
        return ' '.join(list(map(lambda kv : f"{kv[0]} = {kv[1]}",results.items())))

    def _format(self, epoch : int, train_results : dict[str, Any], val_results : dict[str, Any]) -> str:

        return f"Epoch {epoch} - " \
            + ','.join(list(map(lambda kv : f"train_{kv[0]} = {kv[1]}",train_results.items()))) + ',' \
            + ','.join(list(map(lambda kv : f"val_{kv[0]} = {kv[1]}",val_results.items()))) + "\n"
    
    def save(self, meta : dict | None = None, filename : str | None = None) -> None:

        epoch = self.start_epoch + self.epochs
        path = os.path.join(self.checkpoints_folder, (f"checkpoint_{epoch}.pt" if filename is None else filename))

        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "epoch": epoch,
            "meta": meta
        }, path)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> tuple[Any, Tensor]:

        ### Put the model in training mode
        self.model.train()

        ### Zero the gradients
        self.optimizer.zero_grad()

        ### Forward pass
        y_pred = self.model(x)
        
        ### Compute the loss
        loss = self.criterion(y_pred, y)

        ### Backward pass
        loss.backward()

        ### Update the weights
        self.optimizer.step()

        ### Update the scheduler
        if self.scheduler is not None and self.scheduler_mode == "step":
            self.scheduler.step()

        return y_pred, loss
    
    def val_step(self, x: torch.Tensor, y: torch.Tensor) -> tuple[Any, Tensor]:
            
        ### Put the model in evaluation mode
        self.model.eval()
    
        ### Forward pass
        with torch.inference_mode():
            
            y_pred = self.model(x)
    
            ### Compute the loss
            loss = self.criterion(y_pred, y)
    
        return y_pred, loss
    
    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
            
        x, y = x.to(self.device), y.to(self.device)
    
        y_pred, loss = self.train_step(x, y)
    
        results = { "loss": loss.item() }
    
        for name, metric in self.metrics.items():
            results[name] = metric(y_pred, y).item()
    
        return results
    
    def val_on_batch(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:

        x, y = x.to(self.device), y.to(self.device)

        y_pred, loss = self.val_step(x, y)

        results = { "loss": loss.item() }

        for name, metric in self.metrics.items():
            results[name] = metric(y_pred, y).item()

        return results
    
    def train_on_loader(self, loader: DataLoader, epoch : int | None = None) -> dict[str, Any]:

        results = BatchResults(["loss"] + list(self.metrics.keys()))

        t = tqdm(loader)
        tic = time.time()

        for x, y in t:

            batch_results = self.train_on_batch(x, y)
            results.update_all(batch_results)

            if epoch is not None:
                t.set_description(f"Epoch {epoch} : {self._format_dict(batch_results)}")

        toc = time.time()

        results = results.compute()
        results["time"] = toc - tic

        if epoch is not None:
            results["epoch"] = epoch

        return results
    
    def val_on_loader(self, loader: DataLoader, epoch : int | None = None) -> dict[str, Any]:
            
        results = BatchResults(["loss"] + list(self.metrics.keys()))

        t = tqdm(loader)
        tic = time.time()
    
        for x, y in t:

            batch_results = self.val_on_batch(x, y)
            results.update_all(batch_results)

            if epoch is not None:
                t.set_description(f"Epoch {epoch} : {self._format_dict(batch_results)}")

        toc = time.time()

        results = results.compute()
        results["time"] = toc - tic

        if epoch is not None:
            results["epoch"] = epoch
    
        return results

    def train(self,train_loader: DataLoader,val_loader: DataLoader, epochs: int) -> None:

        self._check()

        best_loss = float('inf')

        for i in range(epochs):

            epoch = self.start_epoch + i + 1
            
            ### Training loop
            train_results = self.train_on_loader(train_loader, epoch)

            print()

            ### Validation loop
            val_results = self.val_on_loader(val_loader, epoch)

            ### Update scheduler
            if self.scheduler is not None and self.scheduler_mode == "epoch":
                self.scheduler.step(val_results["loss"])

            ### Update & Save history
            self.history.update(train_results, "train")
            self.history.update(val_results, "val")

            ### Save checkpoint
            save_every = self.save_every is not None and (epoch + 1) % self.save_every == 0
            save_best = self.save_best and val_results["loss"] < best_loss

            ### Increase total epochs
            self.epochs += 1

            if save_best:
                best_loss = val_results["loss"]

            if save_best or save_every:
                self.save(meta={"save_every": save_every, "save_best": save_best})

            ### Print epoch summary
            print(self._format(epoch, train_results, val_results))

    def load(self, checkpoint: dict[str, Any] | None) -> None:

        if checkpoint is None:
            return

        msg = self.model.load_state_dict(checkpoint["model"])
        print(f"Model loaded with message: {msg}")

        msg = self.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Optimizer loaded with message: {msg}")

        if self.scheduler is not None and checkpoint["scheduler"] is not None:
            msg = self.scheduler.load_state_dict(checkpoint["scheduler"])
            print(f"Scheduler loaded with message: {msg}")

        self.start_epoch = checkpoint["epoch"]