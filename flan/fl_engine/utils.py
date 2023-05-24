from dataclasses import dataclass


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