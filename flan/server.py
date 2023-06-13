from dataclasses import dataclass
from .utils.cache import FileCache, CacheArgs
from .fl_engine.utils import ServerArgs
from .preprocess.fed_qc import FedVariantQCServer


@dataclass
class ServerAncestryArgs:
    cache: CacheArgs
    fed_qc: ServerArgs


class AncestryServer:
    def __init__(self, args: ServerAncestryArgs) -> None:
        self.args = args
        self.cache = FileCache(args.cache)
        self.variant_qc_server = FedVariantQCServer(args.variant_qc_server) 
    
    def prepare(self) -> None:
        print(f'starting to manage data preparation on the server side')
        self.variant_qc_server.fit_transform()