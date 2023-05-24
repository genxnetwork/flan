from flan import GlobalAncestry, GlobalArgs, NodeAncestry, NodeAncestryArgs, AncestryServer, ServerAncestryArgs
from flan.bin import linux
import subprocess
import sys
import os
import logging
import hydra
from hydra.utils import instantiate


def flan():
    if len(sys.argv) < 2:
        raise ValueError(f'Please use one of global, server, client working modes')
    
    mode = sys.argv[1]
    if mode == 'global':
        global_ancestry()
    elif mode == 'server':
        server_ancestry()
    elif mode == 'client':  
        node_ancestry()
    else:
        raise ValueError(f'Please use one of global, server, client working modes and not {mode}')
    

def global_ancestry():
    print('current working directory is ', os.getcwd())
    hydra.initialize(version_base='1.3', config_path = 'configs')
    
    if len(sys.argv) < 3:
        raise ValueError(f'Please use one of prepare,fit,predict commands')    
    cmd = sys.argv[2]
    print(sys.argv[3:])
    conf = hydra.compose('config.yaml', sys.argv[3:])
    args = instantiate(conf)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    ga = GlobalAncestry(args)
    if cmd == 'prepare':
        ga.prepare()
    elif cmd == 'fit':
        ga.fit()
    elif cmd == 'predict':
        ga.prepare_for_prediction()
        ga.predict()
    else:
        raise ValueError(f'Please use one of prepare,fit,predict commands')
    
    
def node_ancestry():
    hydra.initialize(version_base='1.3', config_path = 'configs')
    
    if len(sys.argv) < 3:
        raise ValueError(f'Please use one of prepare,fit,predict commands')    
    cmd = sys.argv[2]
    print(sys.argv[3:])
    conf = hydra.compose('node.yaml', sys.argv[3:])
    args = instantiate(conf)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    ga = NodeAncestry(args)
    if cmd == 'prepare':
        ga.prepare()
    else:
        raise ValueError(f'Please use one of prepare commands')
    
    
def server_ancestry():
    hydra.initialize(version_base='1.3', config_path = 'configs')
    
    if len(sys.argv) < 3:
        raise ValueError(f'Please use one of prepare,fit,predict commands')    
    cmd = sys.argv[2]
    print(sys.argv[3:])
    
    conf = hydra.compose('server.yaml', sys.argv[3:])
    args = instantiate(conf)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    ga = AncestryServer(args)
    if cmd == 'prepare':
        ga.prepare()
    else:
        raise ValueError(f'Please use one of prepare commands')    
    
    
def plink2():
    path = linux.__path__[0]
    command = path + "/plink2"

    result = subprocess.run([command] + sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result.stderr.decode('utf-8'))
    print(result.stdout.decode('utf-8'))

    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode('utf-8'))
    
    
    