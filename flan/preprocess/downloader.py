from dataclasses import dataclass
from typing import Optional
import pandas
from urllib.request import urlretrieve
import logging
from tqdm import tqdm
from pathlib import Path

from ..utils.cache import FileCache
from ..utils.plink import run_plink


@dataclass
class SourceArgs:
    link: Optional[str] = None


TG_SUPERPOP_DICT = {'ACB': 'AFR', 'ASW': 'AFR', 'ESN': 'AFR', 'GWD': 'AFR', 'LWK': 'AFR', 'MSL': 'AFR', 'YRI': 'AFR', 
                    'CLM': 'AMR', 'MXL': 'AMR', 'PEL': 'AMR', 'PUR': 'AMR', 
                    'CDX': 'EAS', 'CHB': 'EAS', 'CHS': 'EAS', 'JPT': 'EAS', 'KHV': 'EAS', 
                    'CEU': 'EUR', 'FIN': 'EUR', 'GBR': 'EUR', 'IBS': 'EUR', 'TSI': 'EUR', 
                    'BEB': 'SAS', 'GIH': 'SAS', 'ITU': 'SAS', 'PJL': 'SAS', 'STU': 'SAS'}


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
    
    def _create_keep_samples_file(self, cache: FileCache):
        sf_path = cache.keep_samples_path()
        phenotypes = pandas.read_table(cache.phenotype_path())
        print(phenotypes.columns)
        to_keep = phenotypes.loc[phenotypes['pop'].isin(TG_SUPERPOP_DICT), ['sample']]
        to_keep.to_csv(sf_path, sep='\t', index=False)
        
    def _convert_to_pfile(self, cache: FileCache):
        run_plink(
            args_list = ['--make-pgen'],
            args_dict = {
                '--vcf': str(cache.vcf()[0]),
                '--keep': str(cache.keep_samples_path()),
                '--out': str(cache.pfile_path())
            }
        )
    
    def _download(self, cache: FileCache) -> None:
        vcf, tbi = cache.vcf()
        
        self._download_file(self.affymetrix_link, vcf)
        self._download_file(self.affymetrix_link + '.tbi', tbi)
        self._download_file(self.panel_link, cache.phenotype_path())
            
        return vcf
    
    def fit_transform(self, cache: FileCache) -> None:
        self._download(cache)
        self._create_keep_samples_file(cache)
        self._convert_to_pfile(cache)


class TGPanelDownloader(TGDownloader):
    def __init__(self, args: SourceArgs):
        self.args = args
        self.panel_link = "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/affy_samples.20141118.panel"

    def _update_psam(self, cache: FileCache) -> None:
        psam = pandas.read_table(cache.pfile_path() + '.psam')
        panel = pandas.read_table(cache.phenotype_path())
        
        psam.loc[:, 'ancestry'] = panel['pop'].map(TG_SUPERPOP_DICT)
        
    
    def fit_transform(self, cache: FileCache) -> None:
        self._download_file(self.panel_link, cache.phenotype_path())
        self._create_keep_samples_file(cache)
        