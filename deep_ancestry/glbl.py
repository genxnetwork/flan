from typing import Dict
import logging
import os
import sys

from .pca import PCA
from .preprocess import QC, TGDownloader


logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


class GlobalAncestry:
    def __init__(self, n_components: int = 10, cache_dir: str = None, qc_config: Dict = None) -> None:
        if cache_dir is None or cache_dir == '':
            cache_dir = os.path.expanduser('~/.cache/deep_ancestry')
        self.tg_downloader = TGDownloader(cache_dir)
        self.temp_pfile = os.path.join(cache_dir, 'temp')
        self.qc = QC(qc_config)
        self.pca = PCA(n_components)
        
    def prepare(self) -> None:
        print(f'Preparing data for global ancestry inference')
        vcf = self.tg_downloader.download()
        print(f'Running QC with {self.qc.qc_config} config')
        self.qc.fit(vcf, self.temp_pfile)
        print(self.temp_pfile)
        print(f'Running PCA with {self.pca.n_components} components')
        self.pca.fit(self.temp_pfile)
        print(f'Global ancestry inference data preparation finished')
    
    def fit(self) -> None:
        pass
    
    def predict(self) -> None:
        pass
    
