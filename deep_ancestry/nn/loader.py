from dataclasses import dataclass
from typing import Tuple, Any
import numpy
import logging
import pandas 

from ..utils.cache import FileCache


@dataclass
class X:
    train: numpy.ndarray
    val: numpy.ndarray
    test: numpy.ndarray

@dataclass
class Y:
    train: numpy.ndarray
    val: numpy.ndarray
    test: numpy.ndarray

    def astype(self, new_type):
        new_y = Y(
            train=self.train.astype(new_type),
            val=self.val.astype(new_type),
            test=self.test.astype(new_type)
        )
        return new_y


def load_phenotype(phenotype_path: str, out_type = numpy.float32, encode = False) -> numpy.ndarray:
    """
    :param phenotype_path: Phenotypes location
    :param out_type: convert to type
    :param encode: whether phenotypes are strings and we want to code them as ints)
    """
    data = pandas.read_table(phenotype_path)
    data = data.iloc[:, -1].values
    if encode:
        _, data = numpy.unique(data, return_inverse=True)
    return data.astype(out_type)


def load_plink_pcs(path, order_as_in_file=None):
    """ Loads PLINK's eigenvector matrix (e.g. to be used as X for TG). If @order_as_in_file is not None,
     reorder rows of the matrix to match (IID-wise) rows of the file """
    df = pandas.read_csv(path, sep='\t').rename(columns={'#IID': 'IID'}).set_index('IID').iloc[:, 2:]

    if order_as_in_file is not None:
        y = pandas.read_csv(order_as_in_file, sep='\t').set_index('IID')
        assert len(df) == len(y)
        df = df.reindex(y.index)

    return df


class LocalDataLoader:
    def __init__(self) -> None:
        self.logger = logging.getLogger()

    def _load_phenotype(self, path: str) -> numpy.ndarray:
        phenotype = load_phenotype(path, out_type=numpy.float32, encode=True)
        print(f'Phenotype dtype is {phenotype.dtype}')
        if numpy.isnan(phenotype).sum() > 0:
            raise ValueError(f'There are {numpy.isnan(phenotype).sum()} nan values in phenotype from {path}')
        else:
            return phenotype

    def load(self, cache: FileCache, fold: int) -> Tuple[X, Y]:

        y_train = self._load_phenotype(cache.phenotype_path(fold, 'train'))
        y_val = self._load_phenotype(cache.phenotype_path(fold, 'val'))
        y_test = self._load_phenotype(cache.phenotype_path(fold, 'test'))
        y = Y(y_train, y_val, y_test)
        
        x = self._load_pcs(cache, fold)
        return x, y

    def _load_pcs(self, cache: FileCache, fold: int) -> X:
        X_train = load_plink_pcs(path=cache.pca_path(fold, 'train', 'sscore'), 
                                 order_as_in_file=cache.phenotype_path(fold, 'train')).values.astype(numpy.float32)
        X_val = load_plink_pcs(path=cache.pca_path(fold, 'val', 'sscore'), 
                               order_as_in_file=cache.phenotype_path(fold, 'val')).values.astype(numpy.float32)
        X_test = load_plink_pcs(path=cache.pca_path(fold, 'test', 'sscore'), 
                                order_as_in_file=cache.phenotype_path(fold, 'test')).values.astype(numpy.float32)
        return X(X_train, X_val, X_test)

