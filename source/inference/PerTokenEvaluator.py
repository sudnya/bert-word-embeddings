from models.Vocab import Vocab

class PerTokenEvaluator:
    def __init__(self, config):
        self.config = config
        self.vocab = Vocab(config)

    def initialize(self):
        self.entropy = 0.0
        self.totalBytes = 0

    def evaluate(self, inputs, labels, predictions):
        import math

        batchSize = predictions.shape[0]
        sequenceLength = predictions.shape[1]

        for batch in range(batchSize):
            for token in range(sequenceLength):
                p = predictions[batch, token, 0]
                tokenBytes = self.vocab.getTokenBytes(token)

                self.entropy += (-math.log(p)) / tokenBytes
                self.totalBytes += tokenBytes

    def getRequestedPredictions(self, inputs, labels):
        import numpy
        return numpy.expand_dims(labels, axis=2)

    def finalize(self):
        return 2**(self.entropy/self.totalBytes)

