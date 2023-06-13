from typing import Dict, Tuple, List
import pandas
from pathlib import Path
import numpy
from time import sleep

from grpc import RpcError
from flwr.server.strategy import FedAvg
from flwr.server import start_server
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
from ..fl_engine.utils import ClientArgs, ServerArgs, NodeArgs

   
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
    def __init__(self, qc_config: Dict, pfile_prefix: str, output_prefix: str, variants_file: str) -> None:
        self.qc_config = qc_config
        self.pfile_prefix = pfile_prefix
        self.output_prefix = output_prefix
        self.variants_file = variants_file

    def get_parameters(self) -> Parameters:
        pass
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        '''
        Reads local variants file and sends it to server for aggregation
        '''
        variants = pandas.read_table(self.variants_file)
        return variants.loc[:, 'ID'].values, len(variants), {}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        '''
        Gets aggregated SNP ids in <parameters> and extracts them from local genotype file
        '''
        variants = pandas.DataFrame(values = parameters[0].reshape(-1, 1), columns=['ID'])
        variants.to_csv(self.variants_file + '.aggregated', sep='\t', index=False, header=False)
        run_plink(args_list=['--make-pgen'], args_dict={
                **{'--pfile': self.pfile_prefix, 
                   '--extract': self.variants_file + '.aggregated', 
                   '--out': self.pfile_prefix},
                **self.qc_config
            } # Merging dicts here
        )
        return 0.0, 0, {}


class FedVariantQCServer:
    def __init__(self, server_args: ServerArgs, qc_config: Dict) -> None:
        self.qc_config = qc_config
        self.args = server_args
        
    def fit_transform(self, cache: FileCache) -> None:
        server = start_server(
            server_address=f'{self.args.host}:{self.args.port}',
            strategy=FedVariantStrategy(freq_path=cache.freq_path, **self.args.strategy_args),
            config={"num_rounds": 1},
            force_final_distributed_eval=True
        )
        print(server)
        
        
class FedVariantQCNode:
    def __init__(self, node_args: NodeArgs) -> None:
        self.args = node_args
        
    def fit_transform(self) -> None:
        for i in range(self.args.connect_iters):
            try:
                print(f'starting numpy client with server {self.args.client.server}')
                start_numpy_client(f'{self.args.client.server}:{self.args.client.port}', self.client)
                return True
            except RpcError as re:
                # probably server has not started yet
                print(re)
                sleep(self.args.connect_timeout)
                continue
            except Exception as e:
                print(e)
                self.logger.error(e)
                raise e
        return False  