
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

        labels = self.addTokenLabels(chunk, documentId)
        inputs = self.maskOffTokens(labels)

        secondLabels = self.addTokenLabels(secondChunk, secondDocumentId)
        secondInputs = self.maskOffTokens(secondLabels)

        return inputs, labels, secondInputs, secondLabels

    def addTokenLabels(self, chunk, documentIds):

        return [documentIds[0]] + list(chunk)

    def maskOffTokens(self, labels):
        inputs = list(labels)

        for i in range(1, len(labels)):
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


