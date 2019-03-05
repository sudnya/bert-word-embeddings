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
                p = probabilities[labels[batch, token]]
                tokenBytes = self.vocab.getBytesPerToken(token)

                self.entropy += (-math.log(p)) / tokenBytes
                self.totalBytes += tokenBytes

    def finalize(self):
        return 2**(self.entropy/self.totalBytes)

