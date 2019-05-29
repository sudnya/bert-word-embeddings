

from models.Vocab import Vocab

import logging

logger = logging.getLogger(__name__)

class TokenizerAdaptor:
    def __init__(self, config, source):
        self.config = config
        self.source = source
        self.buffer = []
        self.idBuffer = []
        self.vocab = self.loadVocab()
        self.maximumSize = None
        self.tokenCount = None

    def loadVocab(self):
        return Vocab(self.config)

    def next(self):
        self.fillBuffer()
        return self.matchBestToken()

    def fillBuffer(self):
        while len(self.buffer) < self.vocab.getMaximumTokenSize():
            character, documentId = self.source.next()
            if len(character) == 0:
                break
            self.buffer.append(character)
            self.idBuffer.append(documentId)

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

        raise ValueError("Could not find token in buffer '" + str(self.buffer) + "'")


    def tryMatchBestToken(self):
        possibleWord = self.buffer[0]
        documentId = self.idBuffer[0]

        match = None


        for i in range(1, len(self.buffer)):
            possibleWord += self.buffer[i]

            if self.vocab.isPrefix(possibleWord):
                continue

            if not self.vocab.contains(possibleWord) and len(possibleWord) > 1:
                match = possibleWord[:-1]
                del self.buffer[:i]
                del self.idBuffer[:i]
                break

        if match is None and self.vocab.contains(possibleWord):
            match = possibleWord
            del self.buffer[:]
            del self.idBuffer[:]

        if match is None:
            return None

        token = self.vocab.getToken(match)
        logger.debug("string: '" + match + "' -> " + str(token) + " (" + str(documentId) + ")")

        return token, documentId

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
                expanded = list(repr(character.encode('unicode-escape')))
                self.buffer = self.buffer[:index] + expanded + self.buffer[index + 1:]
                self.idBuffer = self.idBuffer[:index] + [self.idBuffer[index] for
                    _ in range(len(expanded))] + self.idBuffer[index + 1:]
                break

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
        self.reset()

        return count

    def shuffleDocuments(self):
        self.source.shuffleDocuments()

    def clone(self):
        return TokenizerAdaptor(self.config, self.source.clone())



