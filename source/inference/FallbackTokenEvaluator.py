
from models.Vocab import Vocab

from data.UnlimitedVocabTokenizerAdaptor import UnlimitedVocabTokenizerAdaptor
from data.StringDataSource import StringDataSource

import numpy
import logging

logger = logging.getLogger(__name__)

class FallbackTokenEvaluator:
    def __init__(self, config):
        self.config = config
        self.vocab = Vocab(config)

    def initialize(self):
        self.perplexityStates = self.createPerplexityStates(self.getBatchSize())

    def evaluate(self, inputs, labels, predictions):
        inputIndices, predictions, vocabProbabilities = self.rewriteSplitTokens(inputs, labels, predictions)

        self.recordPredictions(predictions, vocabProbabilities, inputIndices, inputs)

    def finalize(self):
        return self.getPerplexity()

    def getBatchSize(self):
        if not "adaptor" in self.config:
            return 1

        if not "batching" in self.config["adaptor"]:
            return 1

        if not "size" in self.config["adaptor"]["batching"]:
            return 1

        return int(self.config["adaptor"]["batching"]["size"])

    def createPerplexityStates(self, count):
        return [PerplexityState(self.vocab) for i in range(count)]

    def getPerplexity(self):
        byteCount = sum([state.getByteCount() for state in self.perplexityStates])
        totalEntropy = sum([state.getEntropy() for state in self.perplexityStates])

        return 2.0 ** (totalEntropy / byteCount)

    def recordPredictions(self, predictions, vocabProbabilities, inputIndices, inputs):
        # predictions is Tensor(batch-size, sequence-length, vocab-size)
        # inputs is Tensor(batch-size, sequence-length)
        batchSize = predictions.shape[0]
        sequenceLength = predictions.shape[1]

        # TODO: replace with something like batch gather
        for batch in range(batchSize):
            for element in range(sequenceLength):
                labelPrediction = predictions[batch, element]
                self.perplexityStates[batch].addPrediction(inputs[batch, :],
                    inputIndices[batch, element], labelPrediction,
                    vocabProbabilities[batch, element, :])

    def rewriteSplitTokens(self, inputs, labels, predictions):
        from functools import reduce

        newInputs = []
        newPredictions = []
        newVocabProbabilities = []

        batchSize = predictions.shape[0]
        sequenceLength = predictions.shape[1]

        # collapse expanded tokens
        for batch in range(batchSize):

            inputString = "".join([self.vocab.getTokenString(token) for token in labels[batch, :] if not Vocab.isReservedToken(token)])
            reservedIndices = set([index for index, token in enumerate(labels[batch, :]) if Vocab.isReservedToken(token)])

            tokenizer = UnlimitedVocabTokenizerAdaptor(StringDataSource(inputString))

            completeTokens = [tokenizer.next() for i in range(tokenizer.size())]

            logger.debug("Reformed input string: '" + str([self.vocab.getTokenString(token) for
                token in labels[batch, :] if not Vocab.isReservedToken(token)]))
            logger.debug( "' tokenized to: " + str(completeTokens))
            logger.debug(" tokens: " + str([self.vocab.getToken(token) for token in completeTokens]))

            index = 0
            completeTokenIndex = 0

            newBatchInputs = []
            newBatchPredictions = []
            newBatchVocabProbabilities = []

            while index < sequenceLength:
                token = labels[batch, index]
                completeToken = completeTokens[completeTokenIndex]

                # get token end
                tokenEndIndex = index + 1
                if self.vocab.getToken(completeToken) != token and not index in reservedIndices:
                    while tokenEndIndex < sequenceLength:
                        possibleToken = labels[batch, tokenEndIndex]
                        if (completeTokenIndex + 1) < len(completeTokens):
                            if self.vocab.getToken(completeTokens[completeTokenIndex + 1]) == possibleToken:
                                break
                        tokenEndIndex += 1
                    if tokenEndIndex > (index + 1):
                        logger.debug("Reformed split tokens: " + str([self.vocab.getTokenString(token)
                            for token in labels[batch, index:tokenEndIndex]]))

                # add token
                newBatchInputs.append([index, tokenEndIndex])
                newBatchVocabProbabilities.append(list(predictions[batch, index, :]))
                newBatchVocabProbabilities[-1][labels[batch, index]] = 0.0

                # compute new probabilities for the merged token
                predictionValues = [predictions[batch, index, label] for label in labels[batch, index:tokenEndIndex]]
                newBatchPredictions.append(reduce(lambda x, y : x * y, predictionValues))

                if not index in reservedIndices:
                    completeTokenIndex += 1

                index = tokenEndIndex

            newInputs.append(newBatchInputs)
            newPredictions.append(newBatchPredictions)
            newVocabProbabilities.append(newBatchVocabProbabilities)

        # pad
        maxLength = max([len(tokens) for tokens in newInputs])

        newInputs = [inputs + [self.getPadToken() for i in range(maxLength - len(inputs))] for inputs in newInputs]
        newPredictions = [predictions + [0.0 for i in range(maxLength - len(predictions))] for predictions in newPredictions]
        newVocabProbabilties = [predictions + [0.0 for i in range(maxLength - len(predictions))] for predictions in newVocabProbabilities]

        return numpy.array(newInputs), numpy.array(newPredictions), numpy.array(newVocabProbabilities)

class PerplexityState:
    def __init__(self, vocab):
        self.byteCount = 0
        self.entropy   = 0.0
        self.vocab     = vocab

    def getByteCount(self):
        return self.byteCount

    def getEntropy(self):
        return self.entropy

    def addPrediction(self, inputTokens, inputIndices, correctLabelPrediction, vocabProbabilities):
        import math

        assert correctLabelPrediction > 0.0, ("Prediction is zero for input tokens: " +
            str(inputTokens) + " with predictions " + str(vocabProbabilities))

        entropy = -math.log(correctLabelPrediction)

        self.entropy += entropy
        self.byteCount += sum([self.vocab.getTokenBytes(inputTokens[index]) for index in range(inputIndices[0], inputIndices[1])])

    def isPredictedToken(self, token):
        return token == Vocab.getMaskToken() or token == Vocab.getVocabOffset()



