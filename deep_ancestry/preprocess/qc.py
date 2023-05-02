from typing import Dict
from ..utils.plink import run_plink


sample_qc_config = {
    '--mind': '0.06',
    '--king-cutoff': '0.0884' # Cutoff corresponds to second degree relatives
}


class QC:
    def __init__(self, qc_config: Dict) -> None:
        self.qc_config = qc_config
        if qc_config is None:
            self.qc_config = sample_qc_config
    
    def fit(self, input_vcf: str, output_pfile: str) -> None:
        run_plink(args_list=['--make-pgen'],
                  args_dict={**{'--vcf': input_vcf, # Merging dicts here
                                '--out': output_pfile,
                                '--memory': '6000',
                                '--set-missing-var-ids': '@:#'},
                             **self.qc_config})
    
    

