
from models.RandomModel import RandomModel
from models.UnigramModel import UnigramModel
from models.NgramModel import NgramModel

class ModelFactory:
    def __init__(self, config, *,
        trainingData=None, validationData=None):

        self.config = config
        self.modelName = config["model"]["type"]
        self.validationData = validationData
        self.trainingData = trainingData

    def create(self):
        if self.modelName == "RandomModel":
            return RandomModel(self.config)

        if self.modelName == "UnigramModel":
            return UnigramModel(self.config, self.trainingData, self.validationData)

        if self.modelName == "NgramModel":
            return NgramModel(self.config, self.trainingData, self.validationData)

        raise RuntimeError("Unknown model name " + self.modelName)

