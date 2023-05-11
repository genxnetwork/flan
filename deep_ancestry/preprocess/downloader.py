from dataclasses import dataclass
from typing import Optional
from urllib.request import urlretrieve
import logging
from tqdm import tqdm
from pathlib import Path

from ..utils.cache import FileCache
from ..utils.plink import run_plink


@dataclass
class SourceArgs:
    link: Optional[str] = None


class TGDownloader:
    def __init__(self, args: SourceArgs) -> None:
        self.args = args
        self.affymetrix_link = "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/ALL.wgs.nhgri_coriell_affy_6.20140825.genotypes_has_ped.vcf.gz"
        self.panel_link = "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/affy_samples.20141118.panel"
        
    def _download_file(self, link: str, output_path: Path) -> None:
        if not output_path.exists():
            with tqdm(total=100, desc='Downloading file', unit='MB') as pbar:
                def reporthook(blocknum, blocksize, totalsize):
                    pbar.update(blocknum * blocksize // 1e+6)
                urlretrieve(link, output_path, reporthook)
                logging.info(f'Downloaded {link} to {output_path}')            
        else:
            logging.info(f'File {output_path} already exists')
                
    def _convert_to_pfile(self, vcf: Path, pfile: Path):
        run_plink(
            args_list = ['--make-pgen'],
            args_dict = {
                '--vcf': str(vcf),
                '--out': str(pfile)
            }
        )
    
    def _download(self, cache: FileCache) -> None:
        vcf, tbi = cache.vcf()
        
        self._download_file(self.affymetrix_link, vcf)
        self._download_file(self.affymetrix_link + '.tbi', tbi)
        self._download_file(self.panel_link, cache.phenotype_path())
            
        return vcf
    
    def fit_transform(self, cache: FileCache) -> None:
        vcf = self._download(cache)
        self._convert_to_pfile(vcf, cache.pfile_path())
