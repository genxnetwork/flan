from typing import Dict, Tuple, List, Optional
import logging
import os
import sys
from torch.nn.functional import cross_entropy
import numpy
import pandas
from dataclasses import dataclass
from pytorch_lightning import Trainer
from tqdm import trange
import mlflow
import plotly.express as px
import plotly.graph_objects as go

from .utils.cache import FileCache, CacheArgs
from .pca import PCA, PCAArgs
from .preprocess import QC, QCArgs, FedVariantQCNode, TGDownloader, FoldSplitter, SplitArgs, SourceArgs, PgenCopy
from .nn.models import MLPClassifier, BaseNet, ModelArgs, OptimizerArgs, SchedulerArgs
from .nn.lightning import X, Y, DataModule
from .nn.loader import LocalDataLoader
from .glbl import TrainArgs
from .fl_engine.utils import NodeArgs


logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


@dataclass
class NodeAncestryArgs:
    name: str
    source: SourceArgs
    cache: CacheArgs
    qc: QCArgs
    fed_qc: NodeArgs


class NodeAncestry:
    def __init__(self, args: NodeAncestryArgs) -> None:
        if args.cache.path is None or args.cache.path == '':
            args.cache.path = os.path.expanduser(f'~/.cache/flan/node_{args.name}')
        
        self.args = args
        args.cache.num_folds = args.split.num_folds
        self.cache = FileCache(args.cache)
        self.source = PgenCopy(args.source)
        self.local_variant_qc = QC(args.qc.variant)
        self.federated_variant_qc = FedVariantQCNode(args.fed_qc)
        
    def prepare(self) -> None:

        print(f'Running variant QC with {self.variant_qc.qc_config} config')
        self.variant_qc.fit_transform(self.cache)
        
        print(f'Running federated variant qc with {self.args.fed_qc} args')
        self.federated_variant_qc.fit_transform(self.cache)
        