from typing import Dict, Tuple, List, Optional
import logging
import os
import sys
from torch.nn.functional import cross_entropy
import numpy
import pandas
from dataclasses import dataclass
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import trange
import mlflow
import plotly.express as px
import plotly.graph_objects as go
import shutil
import torch
from pathlib import Path

from .utils.cache import FileCache, CacheArgs
from .pca import PCA, PCAArgs
from .preprocess import QC, TGDownloader, FoldSplitter, SplitArgs, SourceArgs
from .preprocess.qc import QCArgs
from .nn.models import MLPClassifier, BaseNet, ModelArgs, OptimizerArgs, SchedulerArgs
from .nn.lightning import X, Y, DataModule, PredDataModule
from .nn.loader import LocalDataLoader


logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


@dataclass
class TrainArgs:
    batch_size: int
    

@dataclass 
class PredArgs:
    pfile: str
    report_dir: str  
    

@dataclass
class GlobalArgs:
    cache: CacheArgs
    source: SourceArgs
    qc: QCArgs
    split: SplitArgs
    pca: PCAArgs
    train: TrainArgs
    predict: PredArgs
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
        self.sample_splitter = FoldSplitter(args.split)
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
        
    def _plot_target_distribution(self, y: Y, fold: int):
        values, counts = numpy.unique(y.train, return_counts=True)
        _, val_counts = numpy.unique(y.val, return_counts=True)
        _, test_counts = numpy.unique(y.test, return_counts=True)
        assert len(counts) == len(val_counts)
        assert len(counts) == len(test_counts)
        
        fig = go.Figure(data=[
            go.Bar(name='Train', x=values, y=counts/counts.sum()),
            go.Bar(name='Val', x=values, y=val_counts/val_counts.sum()),
            go.Bar(name='Test', x=values, y=test_counts/test_counts.sum())
        ])
        # Change the bar mode
        fig.update_layout(barmode='group')
        with open(self.cache.target_plot_path(fold, 'train'), 'wb') as file:
            file.write(fig.to_image('png'))
    
    def _load_data(self, fold: int) -> Tuple[X, Y]:
        
        x, y = self.data_loader.load(self.cache, fold)
        
        train_stds, val_stds, test_stds = x.train.std(axis=0), x.val.std(axis=0), x.test.std(axis=0)
        for part, stds in zip(['train', 'val', 'test'], [train_stds, val_stds, test_stds]):
            print(f'{part} stds: {numpy.array2string(stds, precision=3, floatmode="fixed")}')

        x.val = x.val * (train_stds / val_stds)
        x.test = x.test * (train_stds / test_stds)
        for part, matrix in zip(['train', 'val', 'test'], [x.train, x.val, x.test]):
            print(f'{part} normalized stds: {numpy.array2string(matrix.std(axis=0), precision=3, floatmode="fixed")}')

        self._plot_target_distribution(y, fold)
        return x, y
    
    def _load_data_for_prediction(self, fold: int) -> numpy.ndarray:
        x, y = self._load_data(fold)
        train_stds = x.train.std(axis=0)
        x_pred, target_names = self.data_loader.load_for_prediction(self.cache)
        pred_stds = x_pred.std(axis=0)
        x_pred = x_pred * (train_stds / pred_stds)
        return x_pred, target_names
    
    def _create_model(self, nclass: int, nfeat: int) -> BaseNet:
        return MLPClassifier(nclass=nclass, nfeat=nfeat,
                             optim_params=self.args.optimizer,
                             scheduler_params=self.args.scheduler,
                             loss=cross_entropy)
    
    def _eval(self, trainer: Trainer, model: BaseNet) -> None:
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
            num_classes = len(numpy.unique(y.train))
            data_module = DataModule(x, y, self.args.train.batch_size)
            
            model = self._create_model(num_classes, x.train.shape[1])
            model_checkpoint = ModelCheckpoint(self.cache.checkpoint_path(fold), filename='{epoch}-{val_loss:.2f}', save_top_k=1, monitor='val_loss')
            trainer = Trainer(max_epochs=self.args.scheduler.epochs_in_round, log_every_n_steps=20, callbacks=[model_checkpoint])
            trainer.fit(model, datamodule=data_module)
            
            shutil.copy(model_checkpoint.best_model_path, self.cache.best_model_path(fold))
            self._eval(trainer, model)  
            mlflow.end_run()
    
    def prepare_for_prediction(self) -> None:
                
        print(f'Preparing data for global ancestry inference')
        
        for ext in ['.pgen', '.pvar', '.psam', '.pvar.zst']:
            if os.path.exists(self.args.predict.pfile + ext):
                shutil.copy(self.args.predict.pfile + ext, str(self.cache.pfile_path(part='pred')) + ext)
        
        
        self.pred_variant_qc = QC({'--extract': str(self.cache.pfile_path(fold_index=0, part='train')) + '.pvar'})
        self.pred_variant_qc.transform(self.cache.pfile_path(part='pred'), self.cache.pfile_path(part='pred'))
        
        # print(f'Running sample QC with {self.sample_qc.qc_config} config')
        # self.sample_qc.fit_transform(self.cache)
        
        self.pca.predict(self.cache)
        print(f'Global ancestry inference data preparation finished')
    
    def predict(self) -> None:
        '''
        Perform global ancestry inference using PCA projection and two-layer MLP.
        Model and PCA training data will be loaded from cache directory
        '''
        model = MLPClassifier.load_from_checkpoint(
            self.cache.best_model_path(fold_index=0),
            optim_params=None, scheduler_params=None, loss=cross_entropy
        )
        x_pred, target_names = self._load_data_for_prediction(0)
        data_module = PredDataModule(x_pred, self.args.train.batch_size)
        trainer = Trainer()
        preds = trainer.predict(model, datamodule=data_module)
        self.report(preds, target_names)
        
    def report(self, preds: numpy.ndarray, target_names: numpy.ndarray) -> None:
        pred_classes = torch.argmax(torch.cat(preds, axis=0), axis=1)
        pred_names = target_names[pred_classes.detach().cpu().numpy()]
        
        psam = pandas.read_table(str(self.cache.pfile_path(part='pred')) + '.psam')
        psam.loc[:, 'Ancestry'] = pred_names
        os.makedirs(self.args.predict.report_dir, exist_ok=True)
        psam.to_csv(Path(self.args.predict.report_dir) / 'predictions.tsv', sep='\t', index=False)
        
    def pc_scatterplot(pca_path: str, scatter_plot_path: str) -> None:
        """ Visualises eigenvector with scatterplot [matrix] """
        eigenvec = pandas.read_table(f'{pca_path}.eigenvec')[['IID', 'PC1', 'PC2']]
        eigenvec = pandas.merge(eigenvec, tg_df, on='IID')
        eigenvec['ethnic_background_name'] = eigenvec['pop'].replace(TG_SUPERPOP_DICT)
        px.scatter(eigenvec, x='PC1', y='PC2', color='split').write_html(scatter_plot_path)
        

