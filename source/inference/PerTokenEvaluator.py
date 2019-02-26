class PerTokenEvaluator:
    def __init__(self, config):
        self.config = config

    def initialize(self):
        self.entropy = 0.0
        self.tokens = 0

    def evaluate(self, inputs, labels, predictions):
        import math

        batchSize = predictions.shape[0]
        sequenceLength = predictions.shape[1]

        for batch in range(batchSize):
            for token in range(sequenceLength):
                probabilities = list(predictions[batch, token, :])
                probabilities[labels[batch, token]] = 1.0 - probabilities[labels[batch, token]]

                self.entropy += sum([-math.log(1.0 - p) for p in probabilities])

        self.tokens += batchSize * sequenceLength

    def finalize(self):
        return 2**(self.entropy/self.tokens)

