from typing import Dict
from dataclasses import dataclass

from ..utils.plink import run_plink
from ..utils.cache import FileCache


@dataclass
class PrunerArgs:
    window: int
    step: int
    r2_threshold: float
    

class Pruner:
    def __init__(self, args: PrunerArgs) -> None:
        self.args = args
    
    def fit_transform(self, cache: FileCache) -> None:
        run_plink(args_list=['--pfile', str(cache.pfile_path()), '--bad-ld', 
                             '--indep-pairwise', str(self.args.window), str(self.args.step), str(self.args.r2_threshold),
                             '--out', str(cache.pfile_path())])
        
        run_plink(args_list=['--pfile', str(cache.pfile_path()),  
                             '--extract', str(cache.pfile_path().with_suffix('.prune.in')),
                             '--make-pgen', '--out', str(cache.pfile_path())])
    
    def transform(self, cache: FileCache, source_path: str, dest_path: str) -> None:
        run_plink(args_list=['--pfile', source_path,
                             '--extract', str(cache.pfile_path().with_suffix('.prune.in')),
                             '--make-pgen', '--out', dest_path])
