from typing import Dict
import logging
import os
import sys

from .utils.cache import FileCache
from .pca import PCA
from .preprocess import QC, TGDownloader, SampleSplitter
from .preprocess.qc import SAMPLE_QC_CONFIG, VARIANT_QC_CONFIG


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

        print(f'Global ancestry inference data preparation finished')
    
    def fit(self) -> None:
        pass
    
    def predict(self) -> None:
        pass
    
