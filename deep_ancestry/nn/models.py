from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy
from pytorch_lightning import LightningModule
import torch
from torch.nn import Linear
from torch.nn.functional import selu
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.entities import Metric
import time

from .lightning import DataModule
from .metrics import ModelMetrics, DatasetMetrics, ClfMetrics


@dataclass
class ModelArgs:
    pass

@dataclass
class OptimizerArgs:
    lr: float = 1e-2
    weight_decay: float = 0.0

@dataclass
class SchedulerArgs:
    gamma: float = 0.9999
    epochs_in_round: int = 1024


@dataclass
class StepMetrics:
    loss: float
    batch_len: int
    raw_loss: float = 0.0
    reg: float = 0.0
    accuracy: float = 0.0


class BaseNet(LightningModule):
    def __init__(self, optim_params: OptimizerArgs, scheduler_params: SchedulerArgs) -> None:
        """Base class for all NN models, should not be used directly

        Args:
            optim_params (OptimizerArgs): Parameters of optimizer
            scheduler_params (SchedulerArgs): Parameters of learning rate scheduler
        """
        super().__init__()
        self.optim_params = optim_params
        self.scheduler_params = scheduler_params
        self.current_round = 1
        self.mlflow_client = MlflowClient()
        self.history: List[Metric] = []
        self.epoch_history: List[StepMetrics] = []
        self.logged_count = 0

    def _add_to_history(self, name: str, value, step: int):
        timestamp = int(time.time() * 1000)
        self.history.append(Metric(name, value, timestamp, step))
        if len(self.history) % 50 == 0:
            self.mlflow_client.log_batch(mlflow.active_run().info.run_id, self.history[-50:])
            self.logged_count = len(self.history)

    def on_train_end(self) -> None:
        unlogged = len(self.history) - self.logged_count
        if unlogged > 0:
            self.mlflow_client.log_batch(mlflow.active_run().info.run_id, self.history[-unlogged:])
            self.logged_count = len(self.history)
        return super().on_train_end()

    def on_predict_end(self) -> None:
        unlogged = len(self.history) - self.logged_count
        if unlogged > 0:
            self.mlflow_client.log_batch(mlflow.active_run().info.run_id, self.history[-unlogged:])
            self.logged_count = len(self.history)
        return super().on_validation_end()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self(x)
        raw_loss = self.calculate_loss(y_hat, y)
        reg = self.regularization()
        loss = raw_loss + reg
        
        self.epoch_history.append(StepMetrics(loss.item(), x.shape[0], raw_loss.detach().item(), reg.detach().item()))
        return loss

    def calculate_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('subclasses of BaseNet should implement loss calculation')

    def regularization(self) -> torch.Tensor:
        raise NotImplementedError('subclasses of BaseNet should implement regularization')

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y_hat, y)
        self.epoch_history.append(StepMetrics(loss.item(), x.shape[0]))
        return loss

    def calculate_avg_epoch_metric(self, metric_list: List[Tuple[float, int]]) -> float:
        total_len = sum(out[1] for out in metric_list)
        avg_loss = sum(out[0]*out[1] for out in metric_list)/total_len
        return avg_loss if isinstance(avg_loss, float) else avg_loss.item()

    def on_train_epoch_end(self) -> None:
        avg_loss = self.calculate_avg_epoch_metric([(eh.loss, eh.batch_len) for eh in self.epoch_history])
        avg_raw_loss = self.calculate_avg_epoch_metric([(eh.raw_loss, eh.batch_len) for eh in self.epoch_history])
        avg_reg = self.calculate_avg_epoch_metric([(eh.reg, eh.batch_len) for eh in self.epoch_history])

        step = self.fl_current_epoch()
        self._add_to_history('train_loss', avg_loss, step)
        self._add_to_history('raw_loss', avg_raw_loss, step)
        self._add_to_history('reg', avg_reg, step)
        self._add_to_history('lr', self.get_current_lr(), step)
        self.log('loss', avg_loss, prog_bar=True)
        self.epoch_history.clear()

    def on_validation_epoch_end(self) -> None:
        avg_loss = self.calculate_avg_epoch_metric([(eh.loss, eh.batch_len) for eh in self.epoch_history])
        self._add_to_history('val_loss', avg_loss, step=self.fl_current_epoch())
        # mlflow.log_metric('val_loss', avg_loss, self.fl_current_epoch())
        self.log('val_loss', avg_loss, prog_bar=True)
        self.epoch_history.clear()

    def fl_current_epoch(self):
        return (self.current_round - 1) * self.scheduler_params.epochs_in_round + self.current_epoch

    def get_current_lr(self):
        if self.trainer is not None:
            optim = self.trainer.optimizers[0]
            lr = optim.param_groups[0]['lr']
        else:
            return self.optim_params['lr']
        return lr

    def set_covariate_weights(self, weights: numpy.ndarray):
        raise NotImplementedError('for this model setting covariate weights is not implemented')

    def _configure_sgd(self):
        last_epoch = (self.current_round - 1) * self.scheduler_params.epochs_in_round if self.scheduler_params is not None else 0

        optimizer = torch.optim.SGD([
            {
                'params': self.parameters(),
                'lr': self.optim_params.lr*self.scheduler_params.gamma**last_epoch if self.scheduler_params is not None else self.optim_params.lr,
                'initial_lr': self.optim_params.lr,
            }], lr=self.optim_params.lr, weight_decay=self.optim_params.weight_decay)

        schedulers = [torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.scheduler_params.gamma, last_epoch=last_epoch
        )] if self.scheduler_params is not None else None
        return [optimizer], schedulers

    def configure_optimizers(self):
        return self._configure_sgd()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return self(batch[0])

    def predict(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        y_pred = []
        y_true = []
        for x, y in loader:
            y_pred.append(self(x).detach().cpu())
            y_true.append(y.cpu())
        return torch.cat(y_pred, dim=0), torch.cat(y_true, dim=0)
    
    def loader_metrics(self, y_hat: torch.Tensor, y: torch.Tensor) -> DatasetMetrics:
        raise NotImplementedError('subclasses of BaseNet should implement loader_metrics')

    def predict_and_eval(self, datamodule: DataModule, **kwargs: Any) -> ModelMetrics:
        raise NotImplementedError('subclasses of BaseNet should implement predict_end_eval')


class MLPClassifier(BaseNet):
    def __init__(self, nclass, nfeat, optim_params, scheduler_params, loss, hidden_size=800, hidden_size2=200, binary=False) -> None:
        super().__init__(optim_params=optim_params, scheduler_params=scheduler_params)
        # self.bn = BatchNorm1d(nfeat)
        self.nclass = nclass
        self.fc1 = Linear(nfeat, hidden_size)
        # self.bn2 = BatchNorm1d(hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size2)
        # self.bn3 = BatchNorm1d(hidden_size2)
        self.fc3 = Linear(hidden_size2, nclass)
        self.loss = loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.bn(x)
        x = selu(self.fc1(x))
        x = selu(self.fc2(x))
        # x = softmax(, dim=1)
        return self.fc3(x)

    def regularization(self):
        return torch.tensor(0)

    def calculate_loss(self, y_hat, y):
        return self.loss(y_hat.squeeze(1), y)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self(x)
        raw_loss = self.calculate_loss(y_hat, y)
        reg = self.regularization()
        loss = raw_loss + reg

        y_pred = torch.argmax(y_hat, dim=1)
        accuracy = (y_pred == y).float().mean()

        self.epoch_history.append(StepMetrics(loss.item(), x.shape[0], raw_loss.detach().item(), reg.detach().item(), accuracy.detach().item()))
        return loss

    def on_train_epoch_end(self) -> None:
        step = self.fl_current_epoch()
        avg_accuracy = self.calculate_avg_epoch_metric([(eh.accuracy, eh.batch_len) for eh in self.epoch_history])
        self._add_to_history('train_accuracy', avg_accuracy, step)
        super(MLPClassifier, self).on_train_epoch_end()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y_hat, y)

        y_pred = torch.argmax(y_hat, dim=1)
        accuracy = (y_pred == y).float().mean()
        self.epoch_history.append(StepMetrics(loss.item(), x.shape[0], accuracy=accuracy.detach().item()))

        return loss

    def on_validation_epoch_end(self) -> None:
        avg_loss = self.calculate_avg_epoch_metric([(eh.loss, eh.batch_len) for eh in self.epoch_history])
        avg_accuracy = self.calculate_avg_epoch_metric([(eh.accuracy, eh.batch_len) for eh in self.epoch_history])
        self._add_to_history('val_loss', avg_loss, step=self.fl_current_epoch())
        self._add_to_history('val_accuracy', avg_accuracy, step=self.fl_current_epoch())
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_accuracy', avg_accuracy)
    
    def loader_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> ClfMetrics:
        loss = self.calculate_loss(y_pred, y_true)
        accuracy = Accuracy(num_classes=self.nclass)
        return ClfMetrics(loss.item(), accuracy(y_pred, y_true).item(), epoch=self.fl_current_epoch(), samples=y_pred.shape[0])
