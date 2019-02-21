
class ModelDescriptionCheckpointer:
    def __init__(self, config, name):
        self.config = config
        self.name = name
        self.version = 0

    def checkpoint(self):
        jsonPath = self.getCheckpointJSONFilePath()
        directory = self.getCheckpointModelDirectoryPath()

        checkpointDescription = {"type" : self.name,
            "path" : directory,
            "version" : (self.version)}

        self.version += 1

        with open(jsonPath, 'w') as jsonFile:
            json.dump(checkpointDescription, jsonFile, indent=4, sort_keys=True)

    def load(self):
        jsonPath = self.getCheckpointJSONFilePath()

        with open(jsonPath) as jsonFile:
            configuration = json.load(jsonFile)

        if not configuration["type"] == self.name:
            raise RuntimeError("Loaded configuration model type '" + configuration["type"] +
                "' does not match expected type '" + self.name + "'")

        self.version = configuration["version"]

    def getCheckpointJSONFilePath(self):
        return os.path.join(self.getExperimentDirectory(), self.config["checkpointPath"])

    def getConfigFilePath(self):
        return os.path.join(self.getExperimentDirectory(), "config.json")

    def getCheckpointModelDirectoryPath(self):
        return os.path.splitext(self.getCheckpointJSONFilePath())[0]

    def getExperimentDirectory(self):
        return self.config["directory"]




