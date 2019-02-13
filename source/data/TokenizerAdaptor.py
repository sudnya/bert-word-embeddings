

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
        # TODO: use a trie
        # try to match the biggest
        for i in range(0, self.maxTokenSize):
            start = self.maxTokenSize - i - 1
            possibleToken = self.buffer[start:]

            if possibleToken in self.vocab:
                del self.buffer[start:]
                return possibleToken

        raise ValueError("Could not find token in buffer '" + self.buffer + "'")






