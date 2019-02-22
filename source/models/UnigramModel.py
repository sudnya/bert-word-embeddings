
from models.Vocab import Vocab
from models.ModelDescriptionCheckpointer import ModelDescriptionCheckpointer

import numpy

import logging

logger = logging.getLogger(__name__)

class UnigramModel:
    def __init__(self, config, trainingData, validationData):
        self.config = config
        self.trainingData = trainingData
        self.validationData = validationData
        self.checkpointer = ModelDescriptionCheckpointer(config, "UnigramModel")

        self.getOrLoadModel()

    def train(self):
        for epoch in range(self.getEpochs()):
            self.runOnTrainingDataset(epoch)

            if self.shouldRunValidation():
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

            print(message, end="\r", flush=True)

        trainEnd = time.time()

        print(message)
        logger.debug(" Training took: " + (str(trainEnd - trainStart)) + " seconds...")

    def trainingStep(self, inputs, labels):
        # just consider the labels
        for batch in range(labels.shape[0]):
            self.totalTokens += labels.shape[1]

            for token in range(labels.shape[1]):
                if not token in self.tokenCounts:
                    self.tokenCounts[token] = 0

                self.tokenCounts[token] += 1

    def runOnValidationDataset(self, epoch):
        import time

        start = time.time()

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

            print(message, end="\r", flush=True)

        end = time.time()

        print(message)
        logger.debug(" Validation took: " + (str(end - start)) + " seconds...")

    def validationStep(self, inputs, labels):
        import math
        crossEntropy = 0.0
        for batch in range(labels.shape[0]):
            for token in range(labels.shape[1]):
                tokenProbability = self.getTokenProbability(token)
                crossEntropy += -math.log(tokenProbability)

        return crossEntropy, labels.shape[0] * labels.shape[1]

    def getTokenProbability(self, token):
        count = 0
        if token in self.tokenCounts:
            count = self.tokenCounts[token]
        # TODO: Implement enhanced good-turing smoothing
        return (count + 1.0) / (self.totalTokens + 1.0)

    def getOrLoadModel(self):
        import os

        self.vocab = Vocab(self.config)

        shouldCreate = not os.path.exists(
            self.checkpointer.getModelDirectory()) or self.shouldCreateModel()

        if shouldCreate:
            self.createModel()
        else:
            self.load()

    def createModel(self):
        self.tokenCounts = numpy.zeros(self.vocab.getSize())
        self.totalTokens = 0

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
        with open(os.path.join(directory, "unigram-statistics.json"), "w") as jsonFile:
            json.dump([self.totalTokens, [i for i in self.tokenCounts]], jsonFile)

        if exists:
            shutil.rmtree(tempDirectory)


    def load(self):
        self.checkpointer.load()

        directory = self.checkpointer.getModelDirectory()

        logger.debug("Loading checkpoint from: " + str(directory))

    def getEpochs(self):
        return int(self.config["model"]["epochs"])

    def getStepsPerEpoch(self):
        return int(self.config["model"]["stepsPerEpoch"])

    def getValidationStepsPerEpoch(self):
        return int(self.config["model"]["validationStepsPerEpoch"])

    def shouldRunValidation(self):
        if not "runValidation" in self.config["model"]:
            return True
        return bool(self.config["model"]["runValidation"])


