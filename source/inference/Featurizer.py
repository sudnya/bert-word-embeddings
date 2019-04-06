
from models.ModelFactory import ModelFactory
from models.Vocab import Vocab
from inference.EvaluatorFactory import EvaluatorFactory

import numpy

import logging

logger = logging.getLogger(__name__)

class Featurizer:
    def __init__(self, config, validationDataset):
        self.config = config
        self.validationDataset = validationDataset

        self.model = self.loadModel()
        self.evaluator = EvaluatorFactory(config).create()

    def featurize(self):
        logger.debug("Running featurizer for " + str(self.getIterations()) + " iterations")

        features = []

        for i in range(self.getIterations()):
            inputs, labels = self.validationDataset.next()

            logger.debug(" sample (inputs: " + str(inputs) + ", label: " + str(labels) + ")")

            features.append(self.model.getFeatures(inputs))

        return numpy.concatenate(features, axis=0)

    def getIterations(self):
        if "iterations" in self.config["predictor"]:
            return int(self.config["predictor"]["iterations"])

        if "validationStepsPerEpoch" in self.config["model"]:
            return int(self.config["model"]["validationStepsPerEpoch"])

        return int(self.config["model"]["validation-steps-per-epoch"])

    def loadModel(self):
        return ModelFactory(self.config).create()


