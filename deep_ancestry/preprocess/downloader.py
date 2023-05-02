import os
from urllib.request import urlretrieve
import logging
from tqdm import tqdm


class TGDownloader:
    def __init__(self, cache_dir: str) -> None:
        self.affymetrix_link = "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/ALL.wgs.nhgri_coriell_affy_6.20140825.genotypes_has_ped.vcf.gz"
        os.makedirs(cache_dir, exist_ok=True)        
        self.cache_dir = cache_dir
    
    def download(self) -> None:
        vcf_gz_file = self.cache_dir + "/affymetrix.vcf.gz"

        with tqdm(total=100, desc='Downloading file', unit='%') as pbar:
            def reporthook(blocknum, blocksize, totalsize):
                pbar.update(int(blocknum * blocksize * 100 / totalsize))

            if not os.path.exists(vcf_gz_file):
                urlretrieve(self.affymetrix_link, vcf_gz_file, reporthook)
                logging.info(f'Downloaded {self.affymetrix_link} to {vcf_gz_file}')            
            else:
                logging.info(f'1000G genotypes file {vcf_gz_file} already exists')
            
            if not os.path.exists(self.cache_dir + "/affymetrix.vcf.gz.tbi"):
                urlretrieve(self.affymetrix_link + '.tbi', vcf_gz_file + ".tbi", reporthook)
                logging.info(f'Downloaded {self.affymetrix_link + ".tbi"} to {vcf_gz_file + ".tbi"}')            
            
        return vcf_gz_file
