from dataclasses import dataclass
from time import sleep
from grpc import RpcError

from flwr.client import NumPyClient, start_numpy_client
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)


@dataclass
class ClientArgs:
    host: str
    port: int
    
    
@dataclass
class ServerArgs:
    host: str
    port: int
    strategy: str
    epochs_in_round: int
    strategy_args: dict = None
    
    
@dataclass
class NodeArgs:
    client: ClientArgs
    connect_iters: int = 20
    connect_timeout: int = 30
    
    
def run_node(node_args: NodeArgs, client: NumPyClient):
    for attempt in range(node_args.connect_iters):
        try:
            address = f'{node_args.client.host}:{node_args.client.port}'
            print(f'attempt {attempt} to start numpy client with server {address}')
            
            start_numpy_client(
                server_address=address, 
                client=client,
                grpc_max_message_length=1536*1024*1024, # 1.5GB
            )
            break
        except RpcError as re:
            # probably server has not started yet
            print(re)
            sleep(node_args.connect_timeout)
            continue
        except Exception as e:
            print(e)
            raise e