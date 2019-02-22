

from models.Vocab import Vocab

import logging

logger = logging.getLogger(__name__)

class TokenizerAdaptor:
    def __init__(self, config, source):
        self.config = config
        self.source = source
        self.buffer = []
        self.vocab = self.loadVocab()

    def loadVocab(self):
        return Vocab(self.config)

    def next(self):
        self.fillBuffer()
        return self.matchBestToken()

    def fillBuffer(self):
        while len(self.buffer) < self.vocab.getMaximumTokenSize():
            self.buffer.append(self.source.next())

    def matchBestToken(self):
        token = self.tryMatchBestToken()
        if not token is None:
            return token

        # expand unicode
        while self.hasUnicode():
            self.expandOneUnicodeCharacter()
            token = self.tryMatchBestToken()
            if not token is None:
                return token

        raise ValueError("Could not find token in buffer '" + self.buffer + "'")


    def tryMatchBestToken(self):
        # try to match the biggest
        for i in range(0, self.vocab.getMaximumTokenSize()):
            end = len(self.buffer) - i - 1
            possibleToken = "".join(self.buffer[:end])
            #logger.debug("trying string: '" + possibleToken + "'")

            if self.vocab.contains(possibleToken):
                del self.buffer[:end]
                token = self.vocab.getToken(possibleToken)
                logger.debug("string: '" + possibleToken + "' -> " + str(token))
                return token

        return None

    def hasUnicode(self):
        for character in self.buffer:
            if self.isUnicode(character):
                return True

        return False

    def isUnicode(self, character):
        return ord(character) > 127

    def expandOneUnicodeCharacter(self):
        for index, character in enumerate(self.buffer):
            if self.isUnicode(character):
                self.buffer = self.buffer[:index] + list(repr(character)) + self.buffer[index + 1:]
                break




