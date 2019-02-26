from models.Vocab import Vocab
from data.UnlimitedVocabTokenizerAdaptor import UnlimitedVocabTokenizerAdaptor

import logging

logger = logging.getLogger(__name__)

class UnkTokenizerAdaptor:
    def __init__(self, config, source):
        self.config = config
        self.source = UnlimitedVocabTokenizerAdaptor(source)
        self.vocab = self.loadVocab()
        self.maximumSize = None
        self.tokenCount = None

    def loadVocab(self):
        return Vocab(self.config)

    def next(self):
        tokenString = self.source.next()

        if self.vocab.contains(tokenString):
            token = self.vocab.getToken(tokenString)
        else:
            token = Vocab.getUnkToken()

        return token

    def size(self):
        if self.tokenCount is None:
            self.tokenCount = self.getTokenCount()
        return self.tokenCount

    def reset(self):
        self.source.reset()

    def setMaximumSize(self, size):
        self.maximumSize = size

    def getTokenCount(self):
        count = 0

        logger.info("Scanning token count...")

        try:
            while True:
                token = self.next()
                count += 1
                if count % 1e6 == 0:
                    logger.info(" " + str(count))

                if not self.maximumSize is None:
                    if count >= self.maximumSize:
                        break
        except ValueError:
            pass

        logger.info("Scanning token count..." + str(count))

        return count




