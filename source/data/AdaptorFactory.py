

from data.TokenizerAdaptor import TokenizerAdaptor
from data.TokenCacheAdaptor import TokenCacheAdaptor

class AdaptorFactor:
    def __init__(self, config):
        self.config = config

    def create(self, source):
        if self.hasTokenizer():
            source = TokenizerAdaptor(self.config, source)

        if self.hasCache():
            source = TokenCacheAdaptor(self.config, source)

        return source

