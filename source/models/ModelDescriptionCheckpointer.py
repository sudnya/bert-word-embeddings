
import os
import shutil

class ModelDescriptionCheckpointer:
    def __init__(self, config, name):
        self.config = config
        self.name = name
        self.version = 0
        self.prefix = ""

    def setPrefix(self, prefix):
        self.prefix=prefix

    def checkpoint(self):
        import json
        jsonPath = self.getCheckpointJSONFilePath()
        directory = self.getCheckpointRoot()
        vocabPath = self.getCheckpointVocabFilePath()
        configPath = self.getConfigFilePath()

        exists = os.path.exists(directory)

        if exists:
            tempDirectory = directory + "-temp"
            shutil.move(directory, tempDirectory)

        os.makedirs(directory)

        checkpointDescription = {"type" : self.name,
            "path" : directory,
            "version" : (self.version)}

        self.version += 1

        with open(jsonPath, 'w') as jsonFile:
            json.dump(checkpointDescription, jsonFile, indent=4, sort_keys=True)

        if vocabPath != self.getCurrentVocabFilePath():
            shutil.copy(self.getCurrentVocabFilePath(), vocabPath)
        elif exists:
            shutil.copy(os.path.join(tempDirectory, "vocab.txt"), vocabPath)

        self.config["model"]["vocab"] = vocabPath

        with open(configPath, 'w') as jsonFile:
            json.dump(self.config, jsonFile, indent=4, sort_keys=True)

    def cleanup(self):
        directory = self.getCheckpointRoot()
        tempDirectory = directory + "-temp"

        if os.path.exists(tempDirectory):
            shutil.rmtree(tempDirectory)

    def load(self):
        import json
        jsonPath = self.getCheckpointJSONFilePath()

        with open(jsonPath) as jsonFile:
            configuration = json.load(jsonFile)

        if not configuration["type"] == self.name:
            raise RuntimeError("Loaded configuration model type '" + configuration["type"] +
                "' does not match expected type '" + self.name + "'")

        self.version = configuration["version"]

    def getCheckpointJSONFilePath(self):
        return os.path.join(self.getCheckpointRoot(), "checkpoint.json")

    def getCheckpointVocabFilePath(self):
        return os.path.join(self.getCheckpointRoot(), "vocab.txt")

    def getCurrentVocabFilePath(self):
        return self.config["model"]["vocab"]

    def getConfigFilePath(self):
        return os.path.join(self.getCheckpointRoot(), "config.json")

    def getModelDirectory(self):
        return os.path.join(self.getExperimentDirectory(), "model")

    def getExperimentDirectory(self):
        return self.config["model"]["directory"]

    def getCheckpointRoot(self):
        if len(self.prefix) == 0:
            return self.getExperimentDirectory()
        else:
            return os.path.join(self.getExperimentDirectory(), self.prefix)





