from typing import Dict
from ..utils.plink import run_plink
from ..utils.cache import FileCache


SAMPLE_QC_CONFIG = {
    '--mind': '0.06',
    '--king-cutoff': '0.0884' # Cutoff corresponds to second degree relatives
}

VARIANT_QC_CONFIG = {
    '--maf': '0.05',
    '--geno': '0.02',
    '--hwe': '0.000001 midp keep-fewhet'
}


class QC:
    def __init__(self, qc_config: Dict) -> None:
        self.qc_config = qc_config
    
    def fit_transform(self, cache: FileCache) -> None:
        run_plink(args_list=['--make-pgen'],
                  args_dict={**{'--pfile': cache.pfile_path(), # Merging dicts here
                                '--out': cache.pfile_path(),
                                '--set-missing-var-ids': '@:#'},
                             **self.qc_config})
    
    

