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
    
    
@dataclass
class NodeArgs:
    client: ClientArgs