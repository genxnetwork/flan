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
    path = linux.__path__[0]
    command = path + "/plink2"

    result = subprocess.run([command] + sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result.stderr.decode('utf-8'))
    print(result.stdout.decode('utf-8'))

    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode('utf-8'))