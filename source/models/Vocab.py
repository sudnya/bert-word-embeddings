
class Vocab:
    def __init__(self, config):
        self.config = config
        self.vocab, self.maxTokenSize = self.loadVocab()

    def loadVocab(self):
        vocab = {}
        maxTokenSize = 0
        with open(self.getVocabPath(), "r") as vocabFile:
            for line in vocabFile:
                if len(line) > 1:
                    string = line[0:-1]
                else:
                    string = line
                vocab[string] = len(vocab) + Vocab.getVocabOffset()
                maxTokenSize = max(maxTokenSize, len(string))

        return vocab, maxTokenSize

    def getVocabPath(self):
        return self.config["model"]["vocab"]

    def contains(self, token):
        return token in self.vocab

    def getMaximumTokenSize(self):
        return self.maxTokenSize

    def getSize(self):
        return len(self.vocab) + Vocab.getVocabOffset()

    def getToken(self, string):
        #print(self.vocab)
        if string in self.vocab:
            return self.vocab[string]
        return Vocab.getUnkToken()
    def getTokenString(self, token):
        for string, tokenId in self.vocab.items():
            if tokenId == token:
                return string

        if token == Vocab.getUnkToken():
            return "<UNK>"

        if token < Vocab.getVocabOffset():
            return "RESERVED_" + str(token)

        raise RuntimeError("invalid token " + str(token))

    @staticmethod
    def getVocabOffset():
        return 3

    @staticmethod
    def getClassLabelToken():
        return 0

    @staticmethod
    def getMaskToken():
        return 1

    @staticmethod
    def getUnkToken():
        return 2

    @staticmethod
    def isReservedToken(token):
        return token < Vocab.getVocabOffset()



