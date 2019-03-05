

from data.TokenizerAdaptor import TokenizerAdaptor
from data.UnkTokenizerAdaptor import UnkTokenizerAdaptor
from data.UnlimitedVocabTokenizerAdaptor import UnlimitedVocabTokenizerAdaptor

from data.CacheAdaptor import CacheAdaptor
from data.ChunkAdaptor import ChunkAdaptor
from data.BatchAdaptor import BatchAdaptor
from data.LabelAdaptor import LabelAdaptor

import logging

logger = logging.getLogger(__name__)

class AdaptorFactory:
    def __init__(self, config):
        self.config = config

    def create(self, source):
        logger.debug("Creating adaptors for dataset " + source.getName())

        if self.hasTokenizer():
            logger.debug(" fallback-tokenizer")
            source = TokenizerAdaptor(self.config, source)
        elif self.hasUnkTokenizer():
            logger.debug(" unk-tokenizer")
            source = UnkTokenizerAdaptor(self.config, source)
        elif self.hasUnlimitedVocabTokenizer():
            logger.debug(" unlimited-vocab-tokenizer")
            source = UnlimitedVocabTokenizerAdaptor(source)

        if self.usesChunks():
            logger.debug(" chunker")
            source = ChunkAdaptor(self.config, source)

        if self.hasCache():
            logger.debug(" cache")
            source = CacheAdaptor(self.config, source)

        if self.hasLabels():
            logger.debug(" labeler")
            source = LabelAdaptor(self.config, source)

        if self.usesBatches():
            logger.debug(" batcher")
            source = BatchAdaptor(self.config, source)

        return source

    def hasTokenizer(self):
        if not "adaptor" in self.config:
            return False

        if not "tokenizer" in self.config["adaptor"]:
            return False

        return True

    def hasUnkTokenizer(self):
        if not "adaptor" in self.config:
            return False

        if not "unk-tokenizer" in self.config["adaptor"]:
            return False

        return True

    def hasUnlimitedVocabTokenizer(self):
        if not "adaptor" in self.config:
            return False

        if not "unlimited-vocab-tokenizer" in self.config["adaptor"]:
            return False

        return True

    def hasCache(self):
        if not "adaptor" in self.config:
            return False

        if not "cache" in self.config["adaptor"]:
            return False

        return True

    def hasLabels(self):
        if not "adaptor" in self.config:
            return False

        if not "labels" in self.config["adaptor"]:
            return False

        return True

    def usesChunks(self):
        if not "adaptor" in self.config:
            return False

        if not "chunking" in self.config["adaptor"]:
            return False

        return True

    def usesBatches(self):
        if not "adaptor" in self.config:
            return False

        if not "batching" in self.config["adaptor"]:
            return False

        return True

