
class Vocab:
    def __init__(self, config):
        self.config = config
        self.vocab, self.tokens, self.maxTokenSize = self.loadVocab()

    def loadVocab(self):
        vocab = {}
        tokens = {}
        maxTokenSize = 0
        with open(self.getVocabPath(), "r") as vocabFile:
            for line in vocabFile:
                if len(line) > 1:
                    string = line[0:-1]
                else:
                    string = line
                if not string in vocab:
                    token = len(vocab) + Vocab.getVocabOffset()
                    vocab[string] = token
                    tokens[token] = string
                    maxTokenSize = max(maxTokenSize, len(string))

        return vocab, tokens, maxTokenSize

    def getVocabPath(self):
        return self.config["model"]["vocab"]

    def save(self, path):
        shutil.copy(self.getVocabPath(), path)

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
        if token == Vocab.getUnkToken():
            return "<UNK>"

        if token < Vocab.getVocabOffset():
            return "RESERVED_" + str(token)

        if token in self.tokens:
            return self.tokens[token]

        raise RuntimeError("invalid token " + str(token))

    def getTokenBytes(self, token):
        return len(self.getTokenString(token).encode('utf-8'))

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



