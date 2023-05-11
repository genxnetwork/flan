from deep_ancestry import GlobalAncestry, GlobalArgs
from deep_ancestry.bin import linux
import subprocess
import sys
import os
import logging
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate


# cs = ConfigStore.instance()
# cs.store(name="config", node=GlobalArgs(None, None, None, None, None, None))
# cs.store(group="cache", node=CacheArgs)


def global_ancestry():
    print('current working directory is ', os.getcwd())
    hydra.initialize(version_base='1.3', config_path = 'configs')
    
    if len(sys.argv) < 2:
        raise ValueError(f'Please use one of prepare,fit,predict commands')    
    cmd = sys.argv[1]
    conf = hydra.compose('config.yaml', sys.argv[2:])
    args = instantiate(conf)    
    print(args)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    ga = GlobalAncestry(args)
    if cmd == 'prepare':
        ga.prepare()
    elif cmd == 'fit':
        ga.fit()
    elif cmd == 'predict':   
        ga.predict()
    else:
        raise ValueError(f'Please use one of prepare,fit,predict commands')
    
    
def plink2():
    path = linux.__path__[0]
    command = path + "/plink2"

    result = subprocess.run([command] + sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result.stderr.decode('utf-8'))
    print(result.stdout.decode('utf-8'))

    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode('utf-8'))
    
    
    