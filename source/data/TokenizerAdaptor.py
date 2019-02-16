

class TokenizerAdaptor:
    def __init__(self, config, source):
        self.config = config
        self.source = source
        self.buffer = []
        self.vocab, self.maxTokenSize = self.loadVocab()

    def next(self):
        self.fillBuffer()
        return self.matchBestToken()

    def loadVocab(self):
        vocab = {}
        with open(self.getVocabPath(), "r") as vocabFile:
            for line in vocabFile:
                vocab[line] = len(vocab)

        return vocab, maxTokenSize

    def fillBuffer(self):
        while len(self.buffer) < self.maxTokenSize:
            self.buffer.append(self.source.getNextCharacter())

    def getVocabPath(self):
        return self.config["model"]["vocab"]

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
        for i in range(0, self.maxTokenSize):
            start = len(self.buffer) - i - 1
            possibleToken = self.buffer[start:]

            if possibleToken in self.vocab:
                del self.buffer[start:]
                return possibleToken

        return None

    def hasUnicode(self):
        for character in self.buffer:
            if self.isUnicode(character):
                return True

        return False

    def isUnicode(self, character):
        return ord(character) > 127

    def expandOneUnicodeCharacter(self):
        for character, index in enumerate(self.buffer):
            if self.isUnicode(character):
                self.buffer = self.buffer[:index] + list(repr(character)) + self.buffer[index + 1:]
                break




