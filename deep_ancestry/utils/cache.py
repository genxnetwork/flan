from typing import Tuple
from pathlib import Path
from dataclasses import dataclass


PCA_EXTENSIONS = {
    'eigenvec': '.eigenvec.allele',
    'counts': '.acount',
    'sscore': '.sscore'
}

@dataclass
class CacheArgs:
    path: Path = Path.home() / '.cache' / 'deep_ancestry'
    num_folds: int = 5


class FileCache:
    def __init__(self, args: CacheArgs) -> None:
        self.root = Path(args.path)
        self.root.mkdir(parents=True, exist_ok=True)
        for subdir in ['ids', 'phenotypes', 'genotypes']:
            (self.root / subdir).mkdir(exist_ok=True)
            for fold in range(args.num_folds):
                (self.root / subdir / f'fold_{fold}').mkdir(exist_ok=True)

        self.num_folds = args.num_folds
            
    def vcf(self) -> Tuple[Path, Path]:
        return self.root / 'affymetrix.vcf.gz', self.root / 'affymetrix.vcf.gz.tbi'
            
    def ids_path(self, fold_index: int = None, part: str = None) -> Path:
        if fold_index is None and part is None:
            return self.root / 'genotypes' /  'genotype.psam'
        elif fold_index is None:
            return self.root / 'ids' / f'{part}_ids.tsv'
        else:
            return self.root / 'ids' / f'fold_{fold_index}' / f'{part}_ids.tsv'
        
    def phenotype_path(self, fold_index: int = None, part: str = None) -> Path:
        if fold_index is None and part is None:
            return self.root / 'phenotypes' /  'phenotypes.tsv'
        elif fold_index is None:
            return self.root / 'phenotypes' / f'{part}_phenotypes.tsv'
        else:
            return self.root / 'phenotypes' / f'fold_{fold_index}' / f'{part}_phenotypes.tsv'
        
    def pfile_path(self, fold_index: int = None, part: str = None) -> Path:
        if fold_index is None and part is None:
            return self.root / 'genotypes' /  'genotype'
        elif fold_index is None:
            return self.root / 'genotypes' / f'{part}_genotype'
        else:
            return self.root / 'genotypes' / f'fold_{fold_index}' / f'{part}_genotype'
        
    def pca_path(self, fold_index: int = None, part: str = None, _type: str = 'eigenvec') -> Path:
        if fold_index is None and part is None:
            return self.root / 'genotypes' /  f'genotype{PCA_EXTENSIONS[_type]}' 
        elif fold_index is None:
            return self.root / 'genotypes' / f'{part}_genotype{PCA_EXTENSIONS[_type]}'
        else:
            return self.root / 'genotypes' / f'fold_{fold_index}' / f'{part}_genotype{PCA_EXTENSIONS[_type]}'
        