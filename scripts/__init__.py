from deep_ancestry import GlobalAncestry
from deep_ancestry.bin import linux
import subprocess
import sys
import logging


def global_ancestry():
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    ga = GlobalAncestry()
    ga.prepare()
    ga.fit()
    ga.predict()
    
    
def plink2():
    # path = linux.__path__[0]
    # command = path + "/plink2"
    subprocess.run(sys.argv)