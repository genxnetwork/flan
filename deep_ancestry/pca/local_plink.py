from ..utils import run_plink


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
    
    def _calculate_allele_frequencies(self, pfile: str) -> None:
        run_plink(args_list=['--freq'], args_dict={'--pfile': pfile, '--out': pfile})

    def fit(self, pfile: str) -> None:
        self._calculate_allele_frequencies(pfile)
        run_plink(args_list=['--pca', str(self.n_components)], args_dict={'--pfile': pfile, '--out': pfile})
    
    def transform(self, pfile: str) -> None:
        pass