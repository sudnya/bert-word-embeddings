
from models.Vocab import Vocab

import numpy

class LabelAdaptor:
    def __init__(self, config, source):
        self.config = config
        self.source = source
        self.secondSource = source.clone()
        self.secondSource.shuffleDocuments()
        self.random = numpy.random.RandomState(seed=self.getSeed())
        self.vocab = Vocab(config)

    def next(self):
        chunk = self.source.next()

        isFromSameSource = self.random.binomial(1, 0.5)

        if isFromSameSource:
            secondChunk = self.source.next()
        else:
            secondChunk = self.secondSource.next()

        chunk, documentId = zip(*chunk)
        secondChunk, secondDocumentId = zip(*secondChunk)

        labels = self.addTokenLabels(chunk, secondChunk, isFromSameSource)
        inputs = self.maskOffTokens(labels)

        return inputs, labels

    def addTokenLabels(self, chunk, secondChunk, isFromSameSource):
        sourceToken = (Vocab.getSameSourceToken() if isFromSameSource else
            Vocab.getDifferentSourceToken())

        return ([sourceToken] + list(chunk) + [Vocab.getSeparatorToken()] +
            list(secondChunk) + [Vocab.getSeparatorToken()])

    def maskOffTokens(self, labels):
        inputs = list(labels)

        size = (len(labels) - 3) // 2

        chunkTokens = list(range(1, size)) + list(range(size + 1, 2 * size + 1))
        for i in chunkTokens:
            if self.random.binomial(1, 0.15):
                if self.random.binomial(1, 0.8):
                    inputs[i] = Vocab.getMaskToken()
                else:
                    if self.random.binomial(1, 0.5):
                        inputs[i] = self.random.randint(Vocab.getVocabOffset(),
                            self.vocab.getSize())

        inputs[0] = Vocab.getClassLabelToken()

        return inputs

    def getSeed(self):
        if not "size" in self.config["adaptor"]["labels"]:
            return 122

        return int(self.config["adaptor"]["labels"]["seed"])

    def reset(self):
        self.random = numpy.random.RandomState(seed=self.getSeed())
        self.source.reset()
        self.secondSource.reset()
        self.secondSource.shuffleDocuments()

    def size(self):
        return self.source.size()

    def setMaximumSize(self, size):
        self.source.setMaximumSize(size)


