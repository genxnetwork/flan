from typing import Dict, Tuple, List, Optional
import logging
import os
import sys
from torch.nn.functional import cross_entropy
import numpy
from dataclasses import dataclass
from pytorch_lightning import Trainer
from tqdm import trange
import mlflow

from .utils.cache import FileCache, CacheArgs
from .pca import PCA, PCAArgs
from .preprocess import QC, TGDownloader, SampleSplitter, SplitArgs, SourceArgs
from .preprocess.qc import QCArgs
from .nn.models import MLPClassifier, BaseNet, ModelArgs, OptimizerArgs, SchedulerArgs
from .nn.lightning import X, Y, DataModule
from .nn.loader import LocalDataLoader


logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


@dataclass
class TrainArgs:
    batch_size: int

@dataclass
class GlobalArgs:
    cache: CacheArgs
    source: SourceArgs
    qc: QCArgs
    split: SplitArgs
    pca: PCAArgs
    train: TrainArgs
    model: ModelArgs = ModelArgs()
    optimizer: OptimizerArgs = OptimizerArgs()
    scheduler: SchedulerArgs = SchedulerArgs()


        

class GlobalAncestry:
    def __init__(self, args: GlobalArgs) -> None:
        if args.cache.path is None or args.cache.path == '':
            args.cache.path = os.path.expanduser('~/.cache/deep_ancestry')
        
        self.args = args
        self.cache = FileCache(args.cache)
        self.tg_downloader = TGDownloader(args.source)
        self.variant_qc = QC(args.qc.variant)
        self.sample_qc = QC(args.qc.sample)
        self.sample_splitter = SampleSplitter(args.split)
        self.pca = PCA(args.pca)
        self.data_loader = LocalDataLoader()
        
    def prepare(self) -> None:

        print(f'Preparing data for global ancestry inference')
        self.tg_downloader.fit_transform(self.cache)
        
        print(f'Running variant QC with {self.variant_qc.qc_config} config')
        self.variant_qc.fit_transform(self.cache)
        
        print(f'Running sample QC with {self.sample_qc.qc_config} config')
        self.sample_qc.fit_transform(self.cache)
        
        print(f'Splitting into train, val and test datasets')
        self.sample_splitter.fit_transform(self.cache)
        
        print(f'Running PCA with {self.pca.args.n_components} components')
        self.pca.fit(self.cache)

        self.pca.transform(self.cache)
        print(f'Global ancestry inference data preparation finished')
    
    def _load_data(self, fold: int) -> Tuple[X, Y]:
        
        x, y = self.data_loader.load(self.cache, fold)
        
        train_stds, val_stds, test_stds = x.train.std(axis=0), x.val.std(axis=0), x.test.std(axis=0)
        for part, stds in zip(['train', 'val', 'test'], [train_stds, val_stds, test_stds]):
            logging.info(f'{part} stds: {numpy.array2string(stds, precision=3, floatmode="fixed")}')

        x.val = x.val * (train_stds / val_stds)
        x.test = x.test * (train_stds / test_stds)
        for part, matrix in zip(['train', 'val', 'test'], [x.train, x.val, x.test]):
            logging.info(f'{part} normalized stds: {numpy.array2string(matrix.std(axis=0), precision=3, floatmode="fixed")}')

        return x, y
    
    def _create_model(self, nclass: int, nfeat: int) -> BaseNet:
        return MLPClassifier(nclass=nclass, nfeat=nfeat,
                             optim_params=self.args.optimizer,
                             scheduler_params=self.args.scheduler,
                             loss=cross_entropy)
    
    def _eval(self) -> None:
        pass
    
    def _start_mlflow_run(self, fold: int):
        mlflow.set_experiment('global_ancestry')
        universal_tags = {
            'model': 'mlp_classifier',
            'phenotype': 'ancestry',
        }
        study_tags = {
            'fold': fold
        }
        self.run = mlflow.start_run(tags=universal_tags | study_tags)
    
    
    def fit(self) -> None:
        for fold in trange(self.cache.num_folds):
            self._start_mlflow_run(fold)
            x, y = self._load_data(fold)
            num_classes = len(numpy.unique(y))
            data_module = DataModule(x, y, self.args.train.batch_size)
            model = self._create_model(num_classes, x.train.shape[1])
            trainer = Trainer(max_epochs=self.args.scheduler.epochs_in_round)
            trainer.fit(model, datamodule=data_module)
            self._eval(trainer, model)  
            mlflow.end_run()      
    
    def predict(self) -> None:
        pass
    
