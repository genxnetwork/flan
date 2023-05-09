from ..utils import run_plink
from ..utils.cache import FileCache
from tqdm import trange


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
    
    def _calculate_allele_frequencies(self, pfile: str) -> None:
        run_plink(args_list=['--freq'], args_dict={'--pfile': pfile, '--out': pfile})

    def fit(self, cache: FileCache) -> None:
        # self._calculate_allele_frequencies(cache.pfile_path())
        for fold in trange(cache.num_folds, desc='PCA on fold', unit='fold'):
            run_plink(args_list=['--pca', 'allele-wts', str(self.n_components)], 
                      args_dict={'--pfile': str(cache.pfile_path(fold, 'train')),
                                 '--freq': 'counts', 
                                 '--out': cache.pfile_path(fold, 'train')})
    
    def transform(self, cache: FileCache) -> None:
        for fold in trange(cache.num_folds, desc='PCA projection on fold', unit='fold'):
            for part in ['val', 'test']:
                run_plink(args_list=['--score', str(cache.pca_path(fold, 'train', 'eigenvec')), 
                                     '2', '5', 
                                     'header-read', 'no-mean-imputation', 'variance-standardize'],
                          args_dict={'--pfile': str(cache.pfile_path(fold, 'train')),
                                     '--read-freq': str(cache.pca_path(fold, 'train', 'counts')),
                                     '--out': cache.pfile_path(fold, part)})