from typing import Dict, Tuple
import logging
import os
import sys
from torch.nn.functional import mse_loss, cross_entropy, binary_cross_entropy
import numpy
from pytorch_lightning import Trainer

from .utils.cache import FileCache
from .pca import PCA
from .preprocess import QC, TGDownloader, SampleSplitter
from .preprocess.qc import SAMPLE_QC_CONFIG, VARIANT_QC_CONFIG
from .nn.models import MLPClassifier, BaseNet
from .nn.lightning import X, Y, DataModule
from .nn.loader import LocalDataLoader, DataLoaderArgs, DataArgs


logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


class GlobalAncestry:
    def __init__(self, n_components: int = 10, cache_dir: str = None, sample_qc_config: Dict = None, variant_qc_config: Dict = None) -> None:
        if cache_dir is None or cache_dir == '':
            cache_dir = os.path.expanduser('~/.cache/deep_ancestry')
        
        self.cache = FileCache(cache_dir)
        self.tg_downloader = TGDownloader()
        self.variant_qc = QC(sample_qc_config if sample_qc_config is not None else SAMPLE_QC_CONFIG)
        self.sample_qc = QC(variant_qc_config if variant_qc_config is not None else VARIANT_QC_CONFIG)
        self.sample_splitter = SampleSplitter()
        self.pca = PCA(n_components)
        self.data_loader = LocalDataLoader(DataLoaderArgs(DataArgs('','',''), DataArgs('', '')))
        
    def prepare(self) -> None:

        print(f'Preparing data for global ancestry inference')
        self.tg_downloader.fit_transform(self.cache)
        
        print(f'Running variant QC with {self.variant_qc.qc_config} config')
        self.variant_qc.fit_transform(self.cache)
        
        print(f'Running sample QC with {self.sample_qc.qc_config} config')
        self.sample_qc.fit_transform(self.cache)
        
        print(f'Splitting into train, val and test datasets')
        self.sample_splitter.fit_transform(self.cache)
        
        print(f'Running PCA with {self.pca.n_components} components')
        self.pca.fit(self.cache)

        self.pca.transform(self.cache)
        print(f'Global ancestry inference data preparation finished')
    
    def _load_data(self) -> Tuple[X, Y]:
        
        x, y = self.data_loader.load()
        
        train_stds, val_stds, test_stds = x.train.std(axis=0), x.val.std(axis=0), x.test.std(axis=0)
        for part, stds in zip(['train', 'val', 'test'], [train_stds, val_stds, test_stds]):
            self.logger.info(f'{part} stds: {numpy.array2string(stds, precision=3, floatmode="fixed")}')

        x.val = x.val * (train_stds / val_stds)
        x.test = x.test * (train_stds / test_stds)
        for part, matrix in zip(['train', 'val', 'test'], [x.train, x.val, x.test]):
            self.logger.info(f'{part} normalized stds: {numpy.array2string(matrix.std(axis=0), precision=3, floatmode="fixed")}')

        return x, y
    
    def _create_model(self, nclass: int, nfeat: int) -> BaseNet:
        return MLPClassifier(nclass=nclass, nfeat=nfeat,
                             optim_params=self.cfg.experiment.optimizer,
                             scheduler_params=self.cfg.experiment.get('scheduler', None),
                             loss=cross_entropy)
    
    def _eval(self) -> None:
        pass
    
    def fit(self) -> None:
        x, y = self._load_data()
        num_classes = len(numpy.unique(y))
        data_module = DataModule(x, y, self.training_params['batch_size'])
        model = self._create_model(num_classes, self.n_components)
        trainer = Trainer(**self.training_params)
        trainer.fit(model, datamodule=data_module)
        self._eval(trainer, model)        
    
    def predict(self) -> None:
        pass
    
