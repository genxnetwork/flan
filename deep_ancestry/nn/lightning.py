from typing import List
from dataclasses import dataclass
import numpy
from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch import tensor


@dataclass
class X:
    train: numpy.ndarray
    val: numpy.ndarray
    test: numpy.ndarray

@dataclass
class Y:
    train: numpy.ndarray
    val: numpy.ndarray
    test: numpy.ndarray

    def astype(self, new_type):
        new_y = Y(
            train=self.train.astype(new_type),
            val=self.val.astype(new_type),
            test=self.test.astype(new_type)
        )
        return new_y


class DataModule(LightningDataModule):
    def __init__(
        self, x: X, y: Y, batch_size: int = None, drop_last: bool = True
    ):
        super().__init__()
        print(f'Train has {x.train.shape[0]} samples, val has {x.val.shape[0]} samples, test has {x.test.shape[0]} samples')
        self.train_dataset = TensorDataset(tensor(x.train), tensor(y.train))
        self.val_dataset = TensorDataset(tensor(x.val), tensor(y.val))
        self.test_dataset = TensorDataset(tensor(x.test), tensor(y.test))
        self.batch_size = batch_size
        self.drop_last = drop_last

    def update_y(self, y: Y):
        assert self.train_dataset.y.shape[0] == y.train.shape[0]
        assert self.val_dataset.y.shape[0] == y.val.shape[0]
        assert self.test_dataset.y.shape[0] == y.test.shape[0]

        self.train_dataset.y = y.train
        self.val_dataset.y = y.val
        self.test_dataset.y = y.test

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=self.drop_last
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=self.drop_last
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=self.drop_last
        )
        return loader

    def predict_dataloader(self) -> List[DataLoader]:
        train_loader = self.train_dataloader()
        val_loader = self.val_dataloader()
        test_loader = self.test_dataloader()
        return [train_loader, val_loader, test_loader]

    def _dataset_len(self, dataset: TensorDataset):
        return len(dataset) // self.batch_size + int(len(dataset) % self.batch_size > 0)

    def train_len(self):
        return len(self.train_dataset)

    def val_len(self):
        return len(self.val_dataset)

    def test_len(self):
        return len(self.test_dataset)

    def feature_count(self):
        return self.train_dataset.feature_count()
