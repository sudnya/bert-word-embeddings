
import logging

logger = logging.getLogger(__name__)

class ChunkAdaptor:
    def __init__(self, config, source):
        self.config = config
        self.source = source

    def next(self):
        chunk = [self.source.next() for i in range(self.getChunkSize())]
        logger.debug("Produced chunk of tokens: " + str(chunk))
        return chunk

    def getChunkSize(self):
        if not "size" in self.config["adaptor"]["chunking"]:
            return 16

        return int(self.config["adaptor"]["chunking"]["size"])

    def size(self):
        return self.source.size() // self.getChunkSize()

    def reset(self):
        self.source.reset()

    def setMaximumSize(self, size):
        self.source.setMaximumSize(size * self.getChunkSize())

    def shuffleDocuments(self):
        self.source.shuffleDocuments()

    def clone(self):
        return ChunkAdaptor(self.config, self.source.clone())



