from typing import Dict, Any, List, Tuple, Optional
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


class BaseNet(LightningModule):
    def __init__(self, optim_params: Dict, scheduler_params: Dict) -> None:
        """Base class for all NN models, should not be used directly

        Args:
            optim_params (Dict): Parameters of optimizer
            scheduler_params (Dict): Parameters of learning rate scheduler
        """
        super().__init__()
        self.optim_params = optim_params
        self.scheduler_params = scheduler_params
        self.current_round = 1
        self.mlflow_client = MlflowClient()
        self.history: List[Metric] = []
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
        return {'loss': loss, 'raw_loss': raw_loss.detach(), 'reg': reg.detach(), 'batch_len': x.shape[0]}

    def calculate_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('subclasses of BaseNet should implement loss calculation')

    def regularization(self) -> torch.Tensor:
        raise NotImplementedError('subclasses of BaseNet should implement regularization')

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y_hat, y)
        return {'val_loss': loss, 'batch_len': x.shape[0]}

    def calculate_avg_epoch_metric(self, outputs: List[Dict[str, Any]], metric_name: str) -> float:
        total_len = sum(out['batch_len'] for out in outputs)
        avg_loss = sum(out[metric_name].item()*out['batch_len'] for out in outputs)/total_len
        return avg_loss if isinstance(avg_loss, float) else avg_loss.item()

    def training_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        avg_loss = self.calculate_avg_epoch_metric(outputs, 'loss')
        avg_raw_loss = self.calculate_avg_epoch_metric(outputs, 'raw_loss')
        avg_reg = self.calculate_avg_epoch_metric(outputs, 'reg')

        step = self.fl_current_epoch()
        self._add_to_history('train_loss', avg_loss, step)
        self._add_to_history('raw_loss', avg_raw_loss, step)
        self._add_to_history('reg', avg_reg, step)
        self._add_to_history('lr', self.get_current_lr(), step)

        '''
        mlflow.log_metrics({
            'train_loss': avg_loss,
            'raw_loss': avg_raw_loss,
            'reg': avg_reg,
            'lr': self.get_current_lr()
        }, step=step)

        mlflow.log_metric('train_loss', avg_loss, self.fl_current_epoch())
        mlflow.log_metric('raw_loss', avg_raw_loss, self.fl_current_epoch())
        mlflow.log_metric('reg', avg_reg, self.fl_current_epoch())
        mlflow.log_metric('lr', self.get_current_lr(), self.fl_current_epoch())
        '''
        # self.log('train_loss', avg_loss)

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        avg_loss = self.calculate_avg_epoch_metric(outputs, 'val_loss')
        self._add_to_history('val_loss', avg_loss, step=self.fl_current_epoch())
        # mlflow.log_metric('val_loss', avg_loss, self.fl_current_epoch())
        self.log('val_loss', avg_loss, prog_bar=True)

    def fl_current_epoch(self):
        return (self.current_round - 1) * self.scheduler_params['epochs_in_round'] + self.current_epoch

    def get_current_lr(self):
        if self.trainer is not None:
            optim = self.trainer.optimizers[0]
            lr = optim.param_groups[0]['lr']
        else:
            return self.optim_params['lr']
        return lr

    def _configure_adamw(self):
        last_epoch = (self.current_round - 1) * self.scheduler_params['epochs_in_round']
        optimizer = torch.optim.AdamW([
            {
                'params': self.parameters(),
                'initial_lr': self.optim_params['lr']/self.scheduler_params['div_factor'],
                'max_lr': self.optim_params['lr'],
                'min_lr': self.optim_params['lr']/self.scheduler_params['final_div_factor']}
            ], lr=self.optim_params['lr'], weight_decay=self.optim_params['weight_decay'])

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self.optim_params['lr'],
                                                        div_factor=self.scheduler_params['div_factor'],
                                                        final_div_factor=self.scheduler_params['final_div_factor'],
                                                        anneal_strategy='linear',
                                                        epochs=int(self.scheduler_params['rounds']*(1.5*self.scheduler_params['epochs_in_round'])+2),
                                                        pct_start=0.1,
                                                        steps_per_epoch=1,
                                                        last_epoch=last_epoch,
                                                        cycle_momentum=False)


        return [optimizer], [scheduler]

    def set_covariate_weights(self, weights: numpy.ndarray):
        raise NotImplementedError('for this model setting covariate weights is not implemented')


    def _configure_sgd(self):
        last_epoch = (self.current_round - 1) * self.scheduler_params['epochs_in_round'] if self.scheduler_params is not None else 0

        optimizer = torch.optim.SGD([
            {
                'params': self.parameters(),
                'lr': self.optim_params['lr']*self.scheduler_params['gamma']**last_epoch if self.scheduler_params is not None else self.optim_params['lr'],
                'initial_lr': self.optim_params['lr'],
            }], lr=self.optim_params['lr'], weight_decay=self.optim_params.get('weight_decay', 0))

        schedulers = [torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.scheduler_params['gamma'], last_epoch=last_epoch
        )] if self.scheduler_params is not None else None
        return [optimizer], schedulers

    def configure_optimizers(self):
        optim_init = {
            'adamw': self._configure_adamw,
            'sgd': self._configure_sgd
        }[self.optim_params['name']]
        return optim_init()

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
        super().__init__(input_size=None, optim_params=optim_params, scheduler_params=scheduler_params)
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

        return {
            'loss': loss,
            'raw_loss': raw_loss.detach(),
            'reg': reg.detach(),
            'batch_len': x.shape[0],
            'accuracy': accuracy
        }

    def training_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        super(MLPClassifier, self).training_epoch_end(outputs)

        step = self.fl_current_epoch()
        avg_accuracy = self.calculate_avg_epoch_metric(outputs, 'accuracy')
        self._add_to_history('train_accuracy', avg_accuracy, step)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y_hat, y)

        y_pred = torch.argmax(y_hat, dim=1)
        accuracy = (y_pred == y).float().mean()

        return {'val_loss': loss, 'val_accuracy': accuracy, 'batch_len': x.shape[0]}

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        avg_loss = self.calculate_avg_epoch_metric(outputs, 'val_loss')
        avg_accuracy = self.calculate_avg_epoch_metric(outputs, 'val_accuracy')
        self._add_to_history('val_loss', avg_loss, step=self.fl_current_epoch())
        self._add_to_history('val_accuracy', avg_accuracy, step=self.fl_current_epoch())
        self.log('val_loss', avg_loss, prog_bar=True)

    def loader_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> ClfMetrics:
        loss = self.calculate_loss(y_pred, y_true)
        accuracy = Accuracy(num_classes=self.nclass)
        return ClfMetrics(loss.item(), accuracy(y_pred, y_true).item(), epoch=self.fl_current_epoch(), samples=y_pred.shape[0])
