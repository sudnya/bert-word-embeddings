

from data.TokenizerAdaptor import TokenizerAdaptor
from data.TokenCacheAdaptor import TokenCacheAdaptor

class AdaptorFactory:
    def __init__(self, config):
        self.config = config

    def create(self, source):
        if self.hasTokenizer():
            source = TokenizerAdaptor(self.config, source)

        if self.hasCache():
            source = TokenCacheAdaptor(self.config, source)

        return source

    def hasTokenizer(self):
        if not "data" in self.config:
            return False

        if not "tokenizer" in self.config:
            return False

        return bool(self.config["data"]["tokenizer"])

    def hasCache(self):
        if not "data" in self.config:
            return False

        if not "cache" in self.config:
            return False

        return bool(self.config["data"]["cache"])

