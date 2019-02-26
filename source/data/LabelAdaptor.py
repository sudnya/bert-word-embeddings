
from models.Vocab import Vocab

import numpy

class LabelAdaptor:
    def __init__(self, config, source):
        self.config = config
        self.source = source
        self.random = numpy.random.RandomState(seed=self.getSeed())
        self.vocab = Vocab(config)

    def next(self):
        chunk = self.source.next()

        labels = self.addTokenLabels(chunk)
        inputs = self.maskOffTokens(labels)

        return inputs, labels

    def addTokenLabels(self, chunk):
        return [Vocab.getClassLabelToken()] + chunk

    def maskOffTokens(self, labels):
        inputs = list(labels)
        for i in range(1, len(inputs)):
            if self.random.binomial(1, 0.15):
                if self.random.binomial(1, 0.8):
                    inputs[i] = Vocab.getMaskToken()
                else:
                    if self.random.binomial(1, 0.5):
                        inputs[i] = self.random.randint(Vocab.getVocabOffset(),
                            self.vocab.getSize())

        return inputs

    def getSeed(self):
        if not "size" in self.config["adaptor"]["labels"]:
            return 122

        return int(self.config["adaptor"]["labels"]["seed"])

    def reset(self):
        return self.source.reset()

    def size(self):
        return self.source.size()

    def setMaximumSize(self, size):
        self.source.setMaximumSize(size)


