from typing import Tuple
from pathlib import Path
from dataclasses import dataclass


PCA_EXTENSIONS = {
    'allele': '.eigenvec.allele',
    'eigenvec': '.eigenvec',
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
        for subdir in ['ids', 'phenotypes', 'genotypes', 'plots']:
            (self.root / subdir).mkdir(exist_ok=True)
            for fold in range(args.num_folds):
                (self.root / subdir / f'fold_{fold}').mkdir(exist_ok=True)

        self.num_folds = args.num_folds
            
    def vcf(self) -> Tuple[Path, Path]:
        return self.root / 'affymetrix.vcf.gz', self.root / 'affymetrix.vcf.gz.tbi'
    
    def keep_samples_path(self) -> Path:
        return self.root / 'keep.samples'
            
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
        
    def pca_path(self, fold_index: int = None, part: str = None, _type: str = 'allele') -> Path:
        if fold_index is None and part is None:
            return self.root / 'genotypes' /  f'genotype{PCA_EXTENSIONS[_type]}' 
        elif fold_index is None:
            return self.root / 'genotypes' / f'{part}_genotype{PCA_EXTENSIONS[_type]}'
        else:
            return self.root / 'genotypes' / f'fold_{fold_index}' / f'{part}_genotype{PCA_EXTENSIONS[_type]}'
        
    def pca_plot_path(self, fold_index: int = None, part: str = None, pc_x: int = 1, pc_y: int = 2) -> Path:
        if fold_index is None and part is None:
            return self.root / 'plots' /  f'pca_pc{pc_x}_pc{pc_y}.html' 
        elif fold_index is None:
            return self.root / 'plots' / f'{part}_pca_pc{pc_x}_pc{pc_y}.html'
        else:
            return self.root / 'plots' / f'fold_{fold_index}' / f'{part}_pca_pc{pc_x}_pc{pc_y}.html'
    
    def target_plot_path(self, fold_index: int = None, part: str = None) -> Path:
        if fold_index is None and part is None:
            return self.root / 'plots' /  f'target.png' 
        elif fold_index is None:
            return self.root / 'plots' / f'{part}_target.png'
        else:
            return self.root / 'plots' / f'fold_{fold_index}' / f'{part}_target.png'
    