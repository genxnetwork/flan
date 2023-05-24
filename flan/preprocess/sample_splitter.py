from dataclasses import dataclass
from itertools import product
import pandas
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from ..utils.cache import FileCache
from ..utils.plink import run_plink


@dataclass
class SplitArgs:
    num_folds: int = 5


class FoldSplitter:
    def __init__(self, args: SplitArgs) -> None:
        self.args = args
        
    def _split_ids(self,
                   cache: FileCache,
                   y: pandas.Series = None, 
                   random_state: int = 34) -> None:
        """
        Splits sample ids from ids_path into K-fold cv. At each fold, 1/Kth goes to test data, 1/Kth (randomly) to val
        and the rest to train. If self.args.num_folds is 1, then 80% of the data goes to train, 10% to val and 10% to test.

        Args:
            cache: FileCache object
            y: y can be passed to trigger StratifiedKFold instead of KFold
            random_state (int): Fixed random_state for train_test_split sklearn function
        """
        if self.args.num_folds == 1:
            train_indices, val_test_indices = train_test_split(indices,
                                                          train_size=0.8,
                                                          random_state=random_state,
                                                          stratify=None if y is None else y.iloc[train_val_indices])
            val_indices, test_indices = train_test_split(val_test_indices,
                                                          train_size=0.5,
                                                          random_state=random_state,
                                                          stratify=None if y is None else y.iloc[train_val_indices])

            for indices, part in zip([train_indices, val_indices, test_indices], ['train', 'val', 'test']):
                out_path = cache.ids_path(fold_index, part)
                ids.iloc[indices, :].to_csv(out_path, sep='\t', index=False)
            
            return None
        
        ids = pandas.read_table(cache.ids_path()).rename(columns={'#IID': 'IID'}).filter(['FID', 'IID'])
        
        if y is None:
            # regular KFold
            kfsplit = KFold(n_splits=self.args.num_folds, shuffle=True, random_state=random_state).split(ids)
        else:
            # stratified KFold, for categorical and possibly binary phenotypes
            y = y.reindex(pandas.Index(ids['IID']))
            kfsplit = StratifiedKFold(n_splits=self.args.num_folds, shuffle=True, random_state=random_state).split(ids, y=y)
            
        for fold_index, (train_val_indices, test_indices) in enumerate(kfsplit):
            train_indices, val_indices = train_test_split(train_val_indices,
                                                          train_size=(self.args.num_folds - 2) / (self.args.num_folds - 1),
                                                          random_state=random_state,
                                                          stratify=None if y is None else y.iloc[train_val_indices])

            for indices, part in zip([train_indices, val_indices, test_indices], ['train', 'val', 'test']):
                out_path = cache.ids_path(fold_index, part)
                ids.iloc[indices, :].to_csv(out_path, sep='\t', index=False)
                
    def _split_genotypes(self, cache: FileCache) -> None:
        for fold_index, part in product(range(cache.num_folds), ['train', 'val', 'test']):
            run_plink(
                args_dict={
                    '--pfile': str(cache.pfile_path()),
                    '--keep': str(cache.ids_path(fold_index, part)),
                    '--out':  str(cache.pfile_path(fold_index, part))
                },
                args_list=['--make-pgen']
            )
    
    def _split_phenotypes(self, cache: FileCache) -> None:
        phenotype = pandas.read_table(cache.phenotype_path(), names=['IID', 'ancestry', 'in_phase3'])
        for fold_index, part in product(range(cache.num_folds), ['train', 'val', 'test']):
            
            ids = pandas.read_table(cache.ids_path(fold_index, part))
            fold_phenotype = phenotype.merge(ids, how='inner', on='IID')[['IID', 'ancestry']]
            fold_phenotype.to_csv(
                cache.phenotype_path(fold_index, part),
                sep='\t', index=False
            )
    
    def fit_transform(self, cache: FileCache) -> None:
        
        self._split_ids(cache)
        self._split_genotypes(cache)
        self._split_phenotypes(cache)