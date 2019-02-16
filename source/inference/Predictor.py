
from models.ModelFactory import ModelFactory

class Predictor:
    def __init__(self, config, validationDataset):
        self.config = config
        self.validationDataset = validationDataset

        self.model = self.loadModel()

    def predict(self):
        modelStates = self.model.createStates(self.getBatchSize())

        perplexityStates = self.createPerplexityStates(self.getBatchSize())

        for i in range(self.getIterations()):
            nextBatch, isDocumentEnds = self.validationDataset.next()

            predictions = model.predict(nextBatch, modelStates)

            perplexityStates.recordPredictions(nextBatch)

            self.resetStates(modelStates, isDocumentEnds)

        return perplexityStates.getPerplexity()

    def loadModel(self):
        return ModelFactory(self.config).create()



