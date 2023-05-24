from dataclasses import dataclass
from .utils.cache import CacheArgs
from .fl_engine.utils import ServerArgs
from .preprocess.fed_qc import FedVariantQCServer
from .preprocess import FileCache


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
        self.variant_qc_server.fit_transform()