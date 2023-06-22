import shutil
import pandas
from ..utils.cache import FileCache

from .downloader import SourceArgs





class PgenCopy:
    def __init__(self, args: SourceArgs) -> None:
        self.args = args
        
    def fit_transform(self, cache: FileCache) -> None:
        for ext in ['.pgen', '.psam', '.pvar.zst']:
            shutil.copy(self.args.link + ext, cache.pfile_path().with_suffix(ext))
            

class PhenotypeExtractor:
    def __init__(self) -> None:
        pass
        
    def fit_transform(self, cache: FileCache) -> None:
        # #IID	PAT	MAT	SEX	SuperPop	Population
        psam = pandas.read_csv(cache.pfile_path().with_suffix('.psam'), sep='\t', header=None, names=['IID', 
                                                                                                      'PAT',
                                                                                                      'MAT',
                                                                                                      'SEX',
                                                                                                      'SuperPop',
                                                                                                      'Population'])
        phenotype = psam.loc[:, ['IID', 'Population']].rename(columns={'Population': 'ancestry'})
        phenotype.loc[:, 'in_phase3'] = 1
        phenotype.to_csv(cache.phenotype_path(), sep='\t', index=False)