
from models.ModelFactory import ModelFactory
from models.Vocab import Vocab

from inference.EvaluatorFactory import EvaluatorFactory

import logging

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, config, validationDataset):
        self.config = config
        self.validationDataset = validationDataset

        self.model = self.loadModel()
        self.evaluator = EvaluatorFactory(config).create()

    def predict(self):
        self.evaluator.initialize()

        logger.debug("Running predictor for " + str(self.getIterations()) + " iterations")

        for i in range(self.getIterations()):
            inputs, labels = self.validationDataset.next()

            logger.debug(" sample (inputs: " + str(inputs) + ", label: " + str(labels) + ")")

            predictions = self.model.predict(inputs)

            self.evaluator.evaluate(inputs, labels, predictions)

        return self.evaluator.finalize()

    def getIterations(self):
        return int(self.config["predictor"]["iterations"])

    def loadModel(self):
        return ModelFactory(self.config).create()


