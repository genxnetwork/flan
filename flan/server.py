from dataclasses import dataclass
from .utils.cache import FileCache, CacheArgs
from .fl_engine.utils import ServerArgs
from .preprocess.fed_qc import FedVariantQCServer
from .pca.federated_plink import FedPCAServer


@dataclass
class ServerAncestryArgs:
    start_stage: str
    cache: CacheArgs
    fed_qc: ServerArgs
    fed_pca: ServerArgs


class AncestryServer:
    def __init__(self, args: ServerAncestryArgs) -> None:
        self.args = args
        self.cache = FileCache(args.cache)
        self.variant_qc_server = FedVariantQCServer(args.fed_qc)
        self.fed_pca_server = FedPCAServer(args.fed_pca)
        
        if args.start_stage is None or args.start_stage == '':
            args.start_stage = 'variant_qc'
            
        self.stages = [
            ('variant_qc', self.variant_qc_server),
            ('pca', self.fed_pca_server)
        ]
        if args.start_stage not in [stage for stage, _ in self.stages]:
            raise ValueError(f'invalid start stage: {args.start_stage}')
        
        stage_start_idx = [i for i, (stage, _) in enumerate(self.stages) if stage == args.start_stage][0]
        self.stages = self.stages[stage_start_idx:]
    
    def prepare(self) -> None:
        for stage, server in self.stages:
            print(f'starting {stage} stage on the server side')
            server.fit_transform(self.cache)
            