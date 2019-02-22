
from models.ModelFactory import ModelFactory
from models.Vocab import Vocab

import numpy
import logging

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, config, validationDataset):
        self.config = config
        self.validationDataset = validationDataset

        self.model = self.loadModel()

    def predict(self):
        perplexityStates = self.createPerplexityStates(self.getBatchSize())

        logger.debug("Running predictor for " + str(self.getIterations()) + " iterations")

        for i in range(self.getIterations()):
            inputs, labels = self.validationDataset.next()

            logger.debug(" sample (inputs: " + str(inputs) + ", label: " + str(labels) + ")")

            predictions = self.model.predict(inputs)

            inputs, predictions = self.rewriteSplitTokens(inputs, labels, predictions)

            self.recordPredictions(perplexityStates, predictions, inputs)

        return self.getPerplexity(perplexityStates)

    def getIterations(self):
        return int(self.config["predictor"]["iterations"])

    def getBatchSize(self):
        if not "adaptor" in self.config:
            return 1

        if not "batching" in self.config["adaptor"]:
            return 1

        if not "size" in self.config["adaptor"]["batching"]:
            return 1

        return int(self.config["adaptor"]["batching"]["size"])

    def createPerplexityStates(self, count):
        return [PerplexityState(self.model.getVocab()) for i in range(count)]

    def loadModel(self):
        return ModelFactory(self.config).create()

    def getPerplexity(self, perplexityStates):
        tokenCount = sum([state.getTokenCount() for state in perplexityStates])
        totalEntropy = sum([state.getEntropy() for state in perplexityStates])

        return 2.0 ** (totalEntropy / tokenCount)

    def recordPredictions(self, perplexityStates, predictions, inputs):
        # predictions is Tensor(batch-size, sequence-length, vocab-size)
        # inputs is Tensor(batch-size, sequence-length)
        batchSize = predictions.shape[0]
        sequenceLength = predictions.shape[1]

        # TODO: replace with something like batch gather
        for batch in range(batchSize):
            for element in range(sequenceLength):
                labelPrediction = predictions[batch, element]
                perplexityStates[batch].addPrediction(inputs[batch, element], labelPrediction)

    def isSingleCharacterToken(self, token):
        if Vocab.isReservedToken(token):
            return True
        vocab = self.model.getVocab()
        return token in set([vocab.getToken(" "), vocab.getToken("\n")])

    def rewriteSplitTokens(self, inputs, labels, predictions):
        from functools import reduce

        newInputs = []
        newPredictions = []

        batchSize = predictions.shape[0]
        sequenceLength = predictions.shape[1]

        # collapse expanded tokens
        for batch in range(batchSize):
            index = 0

            newBatchInputs = []
            newBatchLabels = []
            newBatchPredictions = []

            while index < sequenceLength:
                token = labels[batch, index]

                # get token end
                tokenEndIndex = index + 1
                if not self.isSingleCharacterToken(token):
                    while tokenEndIndex < sequenceLength:
                        if self.isSingleCharacterToken(labels[batch, tokenEndIndex]):
                            break
                        tokenEndIndex += 1
                    if tokenEndIndex > (index + 1):
                        logger.debug("Reformed split tokens: " + str([self.model.vocab.getTokenString(token)
                            for token in labels[batch, index:tokenEndIndex]]))

                # add token
                newBatchInputs.append([index, tokenEndIndex])

                # compute new probabilities for the merged token
                predictionValues = [predictions[batch, index, label] for label in labels[batch, index:tokenEndIndex]]
                newBatchPredictions.append(reduce(lambda x, y : x * y, predictionValues))

                index = tokenEndIndex

            newInputs.append(newBatchInputs)
            newPredictions.append(newBatchPredictions)

        # pad
        maxLength = max([len(tokens) for tokens in newInputs])

        newInputs = [inputs + [self.getPadToken() for i in range(maxLength - len(inputs))] for inputs in newInputs]
        newPredictions = [predictions + [0.0 for i in range(maxLength - len(predictions))] for predictions in newPredictions]

        return numpy.array(newInputs), numpy.array(newPredictions)

class PerplexityState:
    def __init__(self, vocab):
        self.tokenCount = 0
        self.entropy    = 0.0
        self.vocab      = vocab

    def getTokenCount(self):
        return self.tokenCount

    def getEntropy(self):
        return self.entropy

    def addPrediction(self, inputTokens, correctLabelPrediction):
        import math

        for inputToken in inputTokens:
            found = False
            if self.isPredictedToken(inputToken):
                found = True
            if found:
                return

        entropy = -math.log(correctLabelPrediction)

        self.entropy += entropy
        self.tokenCount += 1

    def isPredictedToken(self, token):
        return token == Vocab.getMaskToken() or token == Vocab.getVocabOffset()

