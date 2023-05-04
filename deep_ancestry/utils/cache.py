from typing import Tuple
from pathlib import Path


class FileCache:
    def __init__(self, root_dir: str = None, num_folds: int = 5) -> None:
        if root_dir is None:
            self.root = Path.home() / '.cache' / 'deep_ancestry'
        else:
            self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        for subdir in ['ids', 'phenotypes', 'genotypes']:
            (self.root / subdir).mkdir(exist_ok=True)
            for fold in range(num_folds):
                (self.root / subdir / f'fold_{fold}').mkdir(exist_ok=True)

        self.num_folds = num_folds
            
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
        