
from models.Vocab import Vocab
from models.ModelDescriptionCheckpointer import ModelDescriptionCheckpointer

import numpy

import logging

logger = logging.getLogger(__name__)

class NgramModel:
    def __init__(self, config, trainingData, validationData):
        self.config = config
        self.trainingData = trainingData
        self.validationData = validationData
        self.checkpointer = ModelDescriptionCheckpointer(config, "NgramModel")

        if not self.trainingData is None:
            self.trainingData.setMaximumSize(int(self.config["model"]["steps-per-epoch"]))

        if not self.validationData is None:
            self.validationData.setMaximumSize(
                int(self.config["model"]["validation-steps-per-epoch"]))

        self.getOrLoadModel()

    def train(self):
        for epoch in range(self.getEpochs()):
            self.trainingData.reset()

            self.runOnTrainingDataset(epoch)

            if self.shouldRunValidation():
                self.trainingData.reset()
                self.runOnValidationDataset(epoch)

            self.checkpoint()

    def runOnTrainingDataset(self, epoch):
        import time
        trainStart = time.time()

        for step in range(self.getStepsPerEpoch()):
            generatorStart = time.time()

            inputs, labels = self.trainingData.next()

            generatorEnd = time.time()

            trainStepStart = time.time()
            self.trainingStep(inputs, labels)
            trainStepEnd = time.time()

            message = ("Epoch (" + str(epoch) + " / " + str(self.getEpochs()) +
                "), Step (" + str(step) + " / " + str(self.getStepsPerEpoch()) +
                "), Generator time: " + ("%.2f" % (generatorEnd - generatorStart)) +
                ", training step time: " + ("%.2f" % (trainStepEnd - trainStepStart)))

            if self.shouldPrintStatus():
                print(message, end="\r", flush=True)

        trainEnd = time.time()

        if self.shouldPrintStatus():
            print(message)
        logger.debug(" Training took: " + (str(trainEnd - trainStart)) + " seconds...")

    def trainingStep(self, inputs, labels):
        # just consider the labels
        for batch in range(labels.shape[0]):
            ngramBuffer = []

            for index in range(labels.shape[1]):
                labelToken = labels[batch, index]
                self.addTokenToNgram(ngramBuffer, labelToken)
                self.recordNgramStatistics(ngramBuffer)

    def addTokenToNgram(self, ngramBuffer, token):
        ngramBuffer.append(token)
        if len(ngramBuffer) > self.getMaximumNgramLength():
            del ngramBuffer[0]

    def recordNgramStatistics(self, ngramBuffer):
        for i in range(len(ngramBuffer)):
            start = len(ngramBuffer) - i - 1
            ngram = tuple(ngramBuffer[start:])
            self.ngramTotalCounts[len(ngram) - 1] += 1

            if not ngram in self.ngramCounts:
                self.ngramCounts[ngram] = 0

                if len(ngram) > 1:
                    prefix = ngram[:-1]
                    if not prefix in self.ngramChildCounts:
                        self.ngramChildCounts[prefix] = 0
                    self.ngramChildCounts[prefix] += 1

            self.ngramCounts[ngram] += 1

            if len(ngram) > 1:
                prefix = ngram[:-1]
                if not prefix in self.ngramPrefixCounts:
                    self.ngramPrefixCounts[prefix] = 0
                self.ngramPrefixCounts[prefix] += 1

    def runOnValidationDataset(self, epoch):
        import time

        start = time.time()

        totalCrossEntropy = 0.0
        totalTokens = 0

        for step in range(self.getValidationStepsPerEpoch()):
            generatorStart = time.time()

            inputs, labels = self.validationData.next()

            generatorEnd = time.time()

            stepStart = time.time()
            crossEntropy, tokens = self.validationStep(inputs, labels)
            stepEnd = time.time()

            message = ("Epoch (" + str(epoch) + " / " + str(self.getEpochs()) +
                "), Step (" + str(step) + " / " + str(self.getValidationStepsPerEpoch()) +
                "), Generator time: " + ("%.2f" % (generatorEnd - generatorStart)) +
                ", validation step time: " + ("%.2f" % (stepEnd - stepStart)) +
                ", loss is " + str(crossEntropy/tokens))

            if self.shouldPrintStatus():
                print(message, end="\r", flush=True)

            totalCrossEntropy += crossEntropy
            totalTokens += tokens

        end = time.time()

        if self.shouldPrintStatus():
            print(message)
        logger.debug(" Validation took: " + (str(end - start)) + " seconds... cross entropy is " +
            str(totalCrossEntropy/totalTokens))

    def validationStep(self, inputs, labels):
        import math
        crossEntropy = 0.0
        byteCount = 0
        for batch in range(labels.shape[0]):
            ngramBuffer = []
            for index in range(labels.shape[1]):
                token = labels[batch, index]
                self.addTokenToNgram(ngramBuffer, token)
                probability = self.getNgramProbability(tuple(ngramBuffer))
                #print(sum(self.getVocabProbabilities(tuple(ngramBuffer))), ngramBuffer,
                #    [self.vocab.getTokenString(token) for token in ngramBuffer], probability)
                crossEntropy += -math.log(probability)
                byteCount += self.vocab.getTokenBytes(token)

        return crossEntropy, byteCount

    def getVocabProbabilities(self, ngram):
        prefix = ngram[0:-1]
        return [self.getNgramProbability(prefix + (token,)) for token in range(self.vocab.getSize())]

    # implements kneser ney smoothing: https://en.wikipedia.org/wiki/Kneser–Ney_smoothing
    def getNgramProbability(self, ngram):
        if len(ngram) == 1:
            return (self.getNgramCount(ngram) + 0.0) / self.ngramTotalCounts[0]

        prefixCount = self.getNgramPrefixCount(ngram)

        if prefixCount == 0:
            discountedProbability = 0.0
            scaleFactor = 1.0
        else:
            count = self.getNgramCount(ngram)
            discountedProbability = (max([count + 0.0 - self.getDiscountValue(), 0.0]) / prefixCount)
            scaleFactor = self.getDiscountValue() * (self.getUniqueNgramPrefixCount(ngram) + 0.0) / prefixCount
            #self.ngramTotalCounts[len(ngram) - 1]

        smallerNgramProbability = self.getNgramProbability(ngram[1:])

        return discountedProbability + (scaleFactor * smallerNgramProbability)

    def getNgramCount(self, ngram):
        if ngram in self.ngramCounts:
            return self.ngramCounts[ngram]

        return 0

    def getNgramPrefixCount(self, ngram):
        prefix = ngram[:-1]
        if prefix in self.ngramPrefixCounts:
            return self.ngramPrefixCounts[prefix]

        return 0

    def getUniqueNgramPrefixCount(self, ngram):
        prefix = ngram[:-1]
        if prefix in self.ngramChildCounts:
            return self.ngramChildCounts[prefix]

        return 0.0

    def getOrLoadModel(self):
        import os

        self.vocab = Vocab(self.config)

        shouldCreate = not os.path.exists(
            self.checkpointer.getModelDirectory()) or self.getShouldCreateModel()

        if shouldCreate:
            self.createModel()
        else:
            self.load()

    def createModel(self):
        self.ngramCounts = { (token,) : 1 for token in range(self.vocab.getSize()) }
        self.ngramChildCounts = {}
        self.ngramPrefixCounts = {}
        self.ngramTotalCounts = [0 for i in range(self.getMaximumNgramLength())]
        self.ngramTotalCounts[0] = self.vocab.getSize()

    def predict(self, inputs, requestedPredictions):
        batchSize = requestedPredictions.shape[0]
        length = requestedPredictions.shape[1]
        outputs = requestedPredictions.shape[2]

        probs = numpy.zeros(requestedPredictions.shape)

        for batch in range(batchSize):
            ngramBuffer = []
            for index in range(length):
                localNgram = list(ngramBuffer)
                for output in range(outputs):
                    token = requestedPredictions[batch, index, output]
                    self.addTokenToNgram(localNgram, token)
                    probs[batch, index] = self.getNgramProbability(tuple(localNgram))

                self.addTokenToNgram(ngramBuffer, inputs[batch, index])

        return probs

    def checkpoint(self):
        import json
        import os
        import shutil

        directory = self.checkpointer.getModelDirectory()
        logger.debug("Saving checkpoint to: " + str(directory))

        self.checkpointer.checkpoint()

        exists = os.path.exists(directory)
        if exists:
            tempDirectory = directory + "-temp"

            shutil.move(directory, tempDirectory)

        os.makedirs(directory)
        with open(os.path.join(directory, "ngram-statistics.json"), "w") as jsonFile:
            json.dump(self.getSerializedData(), jsonFile)

        if exists:
            shutil.rmtree(tempDirectory)

    def getSerializedData(self):
        return [self.ngramTotalCounts,
            self.keysToStrings(self.ngramChildCounts),
            self.keysToStrings(self.ngramPrefixCounts),
            self.keysToStrings(self.ngramCounts)]

    def keysToStrings(self, dictionary):
        return {str(k) : v for k,v in dictionary.items()}

    def load(self):
        import os
        import json

        self.checkpointer.load()

        directory = self.checkpointer.getModelDirectory()

        logger.debug("Loading checkpoint from: " + str(directory))
        with open(os.path.join(directory, "ngram-statistics.json"), "r") as jsonFile:
            ngramTotalCounts, ngramChildCounts, ngramPrefixCounts, ngramCounts = json.load(jsonFile)

            self.ngramTotalCounts = ngramTotalCounts
            self.ngramChildCounts = self.keysToTuples(ngramChildCounts)
            self.ngramPrefixCounts = self.keysToTuples(ngramPrefixCounts)
            self.ngramCounts = self.keysToTuples(ngramCounts)

    def keysToTuples(self, dictionary):
        from ast import literal_eval as make_tuple
        return {make_tuple(key) : value for key, value in dictionary.items()}

    def getVocab(self):
        return self.vocab

    def getEpochs(self):
        return int(self.config["model"]["epochs"])

    def getShouldCreateModel(self):
        if not "create-new-model" in self.config["model"]:
            return False
        return bool(self.config["model"]["create-new-model"])

    def getStepsPerEpoch(self):
        return min(int(self.config["model"]["steps-per-epoch"]), self.trainingData.size())

    def getValidationStepsPerEpoch(self):
        return min(int(self.config["model"]["validation-steps-per-epoch"]), self.validationData.size())

    def shouldRunValidation(self):
        if not "runValidation" in self.config["model"]:
            return True
        return bool(self.config["model"]["runValidation"])

    def getMaximumNgramLength(self):
        return int(self.config["model"]["max-ngram-length"])

    def getDiscountValue(self):
        return float(self.config["model"]["discount-value"])

    def shouldPrintStatus(self):
        if "should-print-status" in self.config["model"]:
            return str(self.config["model"]["should-print-status"]) == "True"

        return True



