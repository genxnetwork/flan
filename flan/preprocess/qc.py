from typing import Dict
from dataclasses import dataclass

from ..utils.plink import run_plink
from ..utils.cache import FileCache


@dataclass
class QCArgs:
    sample: Dict[str, str]
    variant: Dict[str, str]


class QC:
    def __init__(self, qc_config: Dict) -> None:
        self.qc_config = qc_config
    
    def fit_transform(self, cache: FileCache) -> None:
        run_plink(args_list=['--make-pgen'],
                  args_dict={**{'--pfile': cache.pfile_path(), # Merging dicts here
                                '--out': cache.pfile_path(),
                                '--set-missing-var-ids': '@:#'},
                             **self.qc_config})
    
    
    def transform(self, source_path: str, dest_path: str) -> None:
        run_plink(args_list=['--make-pgen', '--pfile', str(source_path)],
                  args_dict={**{'--out': str(dest_path),
                                '--set-missing-var-ids': '@:#'},
                             **self.qc_config})
    
