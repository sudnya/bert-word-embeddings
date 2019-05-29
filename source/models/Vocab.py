
import pygtrie

class Vocab:
    def __init__(self, config):
        self.config = config
        self.vocab, self.vocabTrie, self.tokens, self.maxTokenSize = self.loadVocab()

    def loadVocab(self):
        vocabTrie = None
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
                    token = len(tokens) + Vocab.getVocabOffset()
                    vocab[string] = token
                    tokens[token] = string
                    maxTokenSize = max(maxTokenSize, len(string))

        return vocab, vocabTrie, tokens, maxTokenSize

    def getVocabPath(self):
        return self.config["model"]["vocab"]

    def save(self, path):
        shutil.copy(self.getVocabPath(), path)
        with open(self.getVocabPath(), "w") as vocabFile:
            for i in len(self.tokens):
                vocabFile.write(self.tokens[i + Vocab.getVocabOffset()] + "\n")

    def contains(self, token):
        return token in self.vocab

    def isPrefix(self, prefix):
        if self.vocabTrie is None:
            self.makeTrie()

        return self.vocabTrie.has_subtrie(prefix)

    def makeTrie(self):
        self.vocabTrie = pygtrie.CharTrie()

        for string, token in self.vocab.items():
            self.vocabTrie[string] = token

    def getMaximumTokenSize(self):
        return self.maxTokenSize

    def getSize(self):
        return len(self.tokens) + Vocab.getVocabOffset()

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
        return 6

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
    def getSameSourceToken():
        return 3

    @staticmethod
    def getDifferentSourceToken():
        return 4

    @staticmethod
    def getSeparatorToken():
        return 5

    @staticmethod
    def isReservedToken(token):
        return token < Vocab.getVocabOffset()



