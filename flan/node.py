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
from .preprocess import (
    QC, QCArgs, FedVariantQCNode, 
    FoldSplitter, SplitArgs, 
    SourceArgs, PgenCopy, PhenotypeExtractor,
    FedVariantQCArgs
)
from .nn.models import MLPClassifier, BaseNet, ModelArgs, OptimizerArgs, SchedulerArgs
from .nn.lightning import X, Y, DataModule
from .nn.loader import LocalDataLoader
from .glbl import TrainArgs
from .fl_engine.utils import NodeArgs
from .pca.federated_plink import FedPCANode, FedPCAClientArgs


logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


@dataclass
class NodeAncestryArgs:
    name: str
    start_stage: str
    source: SourceArgs
    cache: CacheArgs
    qc: QCArgs
    fed_qc: FedVariantQCArgs
    fed_pca: FedPCAClientArgs
    node: NodeArgs


class NodeAncestry:
    def __init__(self, args: NodeAncestryArgs) -> None:
        if args.cache.path is None or args.cache.path == '':
            args.cache.path = os.path.expanduser(f'~/.cache/flan/node_{args.name}')

        if args.start_stage is None or args.start_stage == '':
            args.start_stage = 'source'
            
        self.args = args
        args.cache.num_folds = 1
        
        self.cache = FileCache(args.cache)
        self.source = PgenCopy(args.source)
        self.phenotype_extractor = PhenotypeExtractor()
        self.local_variant_qc = QC(args.qc.variant)
        self.federated_variant_qc = FedVariantQCNode(args.fed_qc, args.node)
        self.local_splitter = FoldSplitter(SplitArgs(num_folds=1))
        self.federated_pca = FedPCANode(args.fed_pca, args.node)
        
        self.stages = [
            ('source', self.source),
            ('phenotype', self.phenotype_extractor),
            ('local_variant_qc', self.local_variant_qc),
            ('federated_variant_qc', self.federated_variant_qc),
            ('local_splitter', self.local_splitter),
            ('federated_pca', self.federated_pca)
        ]
        
        if args.start_stage not in [stage for stage, _ in self.stages]:
            raise ValueError(f'invalid start stage: {args.start_stage}')
        
        stage_start_idx = [i for i, (stage, _) in enumerate(self.stages) if stage == args.start_stage][0]
        self.stages = self.stages[stage_start_idx:]

        
    def prepare(self) -> None:
        
        for stage, node in self.stages:        
            print(f'running stage {stage} from {self.stages}')
            node.fit_transform(self.cache)
            