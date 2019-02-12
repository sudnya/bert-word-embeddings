
from models.RandomModel import RandomModel

class ModelFactory:
    def __init__(self, config, *, modelName="RandomModel",
        trainingData=None, validationData=None):

        self.config = config
        self.modelName = modelName
        self.validationData = validationData
        self.trainingData = trainingData

    def create(self):
        if self.modelName == "RandomModel":
            return RandomModel(self.config)

        raise RuntimeError("Unknown model name " + self.modelName)

