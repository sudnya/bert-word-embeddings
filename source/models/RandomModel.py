
from models.Vocab import Vocab

import numpy

class RandomModel:
    def __init__(self, config):
        self.config = config
        self.vocab = Vocab(config)

    def train(self):
        # no training happens in this model
        pass


    def predict(self, inputs):
        # output is [batch-size, sequence-length, vocab-size] of 1.0/vocab-size
        batchSize = inputs.shape[0]
        sequenceLength = inputs.shape[1]
        vocabSize = self.getVocabSize()

        return numpy.full([batchSize, sequenceLength, vocabSize], 1.0/vocabSize)

    def getVocabSize(self):
        return self.vocab.getSize()

    def getVocab(self):
        return self.vocab




