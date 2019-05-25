
from models.ModelFactory import ModelFactory
from models.Vocab import Vocab

import numpy
import time

import logging

logger = logging.getLogger(__name__)

class Featurizer:
    def __init__(self, config, validationDataset):
        self.config = config
        self.validationDataset = validationDataset

        self.model = self.loadModel()

    def featurizeOneBatch(self, *, reportTime=False):
        dataStart = time.time()
        inputs, labels, secondInputs, _ = self.validationDataset.next()
        dataEnd = time.time()

        logger.debug(" sample (inputs: " + str(inputs) + ", label: " + str(labels) + ")")

        featureStart = time.time()
        features = self.model.getFeatures(inputs, secondInputs)
        featureEnd = time.time()

        if reportTime:
            return inputs, labels, features, dataEnd - dataStart, featureEnd - featureStart
        else:
            return inputs, labels, features

    def getIterations(self):
        if "iterations" in self.config["predictor"]:
            return int(self.config["predictor"]["iterations"])

        if "validationStepsPerEpoch" in self.config["model"]:
            return int(self.config["model"]["validationStepsPerEpoch"])

        return int(self.config["model"]["validation-steps-per-epoch"])

    def loadModel(self):
        return ModelFactory(self.config).create()


