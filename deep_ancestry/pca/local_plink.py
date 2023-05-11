from dataclasses import dataclass
from tqdm import trange
import pandas
import plotly.express as px

from ..utils import run_plink
from ..utils.cache import FileCache
from ..preprocess.downloader import TG_SUPERPOP_DICT


@dataclass
class PCAArgs:
    n_components: int



class PCA:
    def __init__(self, args: PCAArgs) -> None:
        self.args = args
    
    def _calculate_allele_frequencies(self, pfile: str) -> None:
        run_plink(args_list=['--freq'], args_dict={'--pfile': pfile, '--out': pfile})

    def fit(self, cache: FileCache) -> None:
        # self._calculate_allele_frequencies(cache.pfile_path())
        for fold in trange(cache.num_folds, desc='PCA on fold', unit='fold'):
            run_plink(args_list=['--pca', 'allele-wts', str(self.args.n_components)], 
                      args_dict={'--pfile': str(cache.pfile_path(fold, 'train')),
                                 '--freq': 'counts', 
                                 '--out': cache.pfile_path(fold, 'train')})
    
    def transform(self, cache: FileCache) -> None:
        for fold in trange(cache.num_folds, desc='PCA projection on fold', unit='fold'):
            for part in ['train', 'val', 'test']:
                run_plink(args_list=['--score', str(cache.pca_path(fold, 'train', 'allele')), 
                                     '2', '5', 
                                     'header-read', 'no-mean-imputation', 'variance-standardize'],
                          args_dict={'--pfile': str(cache.pfile_path(fold, part)),
                                     '--read-freq': str(cache.pca_path(fold, 'train', 'counts')),
                                     '--score-col-nums': f'6-{6+self.args.n_components - 1}',
                                     '--out': cache.pfile_path(fold, part)})
                
                self.pc_scatterplot(cache, fold, part)
                
                
    def pc_scatterplot(self, cache: FileCache, fold: int, part: str) -> None:
        """ Visualises eigenvector with scatterplot [matrix] """
        eigenvec = pandas.read_table(cache.pca_path(fold, part, 'sscore'))[['#IID', 'PC1_AVG', 'PC2_AVG']]
        tg_df = pandas.read_table(cache.phenotype_path(fold, part))
        eigenvec = pandas.merge(eigenvec, tg_df, left_on='#IID', right_on='IID')
        print(eigenvec.columns)
        eigenvec['ethnic_background_name'] = eigenvec['ancestry'].replace(TG_SUPERPOP_DICT)
        px.scatter(eigenvec, x='PC1_AVG', y='PC2_AVG', color='ethnic_background_name').write_html(cache.pca_plot_path(fold, part))
