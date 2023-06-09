from typing import Dict, Tuple, List
import pandas
from pathlib import Path
import numpy
from time import sleep
from dataclasses import dataclass

from grpc import RpcError
from flwr.server.strategy import FedAvg
from flwr.server import start_server, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.client import NumPyClient, start_numpy_client
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)  

from ..utils.plink import run_plink
from ..utils.cache import FileCache
from ..fl_engine.utils import ClientArgs, ServerArgs, NodeArgs, run_node

   
class FedVariantStrategy(FedAvg):
    def __init__(self, freq_path: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self.freq_path = freq_path
        
    def aggregate_fit(self, 
                      server_round: int, 
                      results: List[Tuple[ClientProxy, FitRes]], 
                      failures: List[Tuple[ClientProxy, FitRes]]) -> Tuple[Parameters, Dict[str, Scalar]]:

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        variants_list = [
            parameters_to_ndarrays(fit_res.parameters)[0] for _, fit_res in results
        ]
        
        intersection = set(variants_list[0])
        for variants in variants_list[1:]:
            intersection.intersection_update(variants)
            
        variants = numpy.array(list(intersection))
        return ndarrays_to_parameters([variants]), {}
    

class FedVariantQCClient(NumPyClient):
    def __init__(self, pfile_prefix: str, output_prefix: str, variants_file: str) -> None:
        self.pfile_prefix = pfile_prefix
        self.output_prefix = output_prefix
        self.variants_file = variants_file

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        '''
        Reads local variants file and sends it to server for aggregation
        '''
        print(f'we are reading variants from {self.variants_file}')
        print(f'{open(self.variants_file,"r").readlines()[:2]}')
        
        variants = pandas.read_table(self.variants_file, comment='#', header=None, names=['CHROM', 'POS', 'ID', 'REF', 'ALT'])
        ids = variants.loc[:, 'ID'].values.astype(numpy.dtype('U32'))
        print(f'total bytesize of variant ids array is {ids.nbytes} bytes for {len(ids)} variants')
        return [variants.loc[:, 'ID'].values.astype(str)], len(variants), {}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        '''
        Gets aggregated SNP ids in <parameters> and extracts them from local genotype file
        '''
        variants = pandas.DataFrame(data=parameters[0].reshape(-1, 1), columns=['ID'])
        variants.to_csv(self.variants_file + '.aggregated', sep='\t', index=False, header=False)
        print(f'we wrote total of {len(variants)} variants to {self.variants_file + ".aggregated"}')
        run_plink(args_list=['--make-pgen'], args_dict={
                **{'--pfile': self.pfile_prefix, 
                   '--extract': self.variants_file + '.aggregated', 
                   '--out': self.output_prefix}
            } 
        )
        return 0.0, len(variants), {}


class FedVariantQCServer:
    def __init__(self, server_args: ServerArgs) -> None:
        self.args = server_args
        
    def fit_transform(self, cache: FileCache) -> None:
        config = ServerConfig(num_rounds=1)
        server = start_server(
            server_address=f'{self.args.host}:{self.args.port}',
            strategy=FedVariantStrategy(freq_path=cache.pfile_path().with_suffix('.freq'), **self.args.strategy_args),
            config=config,
            # we need it because of the large number of variants, for example 1000G dataset has at least 10M variants
            # each variant id is at most 128 bytes, so 10M * 128 = 1.28GB
            grpc_max_message_length=1536*1024*1024, # 1.5GB
            # force_final_distributed_eval=True
        )
        print(server)
        
        
@dataclass  
class FedVariantQCArgs:
    variant: Dict[str, str]
        
        
class FedVariantQCNode:
    def __init__(self, args: FedVariantQCArgs, node_args: NodeArgs) -> None:
        self.args = args
        self.node_args = node_args
        
    def fit_transform(self, cache: FileCache) -> None:
        self.client = FedVariantQCClient(
            str(cache.pfile_path()),
            str(cache.pfile_path()),
            str(cache.pfile_path().with_suffix('.pvar'))
        )
        run_node(self.node_args, self.client)