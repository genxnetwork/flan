import shutil
from ..utils.cache import FileCache

from .downloader import SourceArgs





class PgenCopy:
    def __init__(self, args: SourceArgs) -> None:
        self.args = args
        
    def fit_transform(self, cache: FileCache) -> None:
        for ext in ['.pgen', '.psam', '.pvar']:
            shutil.copy(self.args.link + ext, cache.path)