
class TextDataSource:
    def __init__(self, config, sourceConfig):
        self.config = config
        self.sourceConfig = sourceConfig
        self.file = open(self.getPath(), encoding='ISO-8859-1')

    def next(self):
        nextCharacter = self.file.read(1)

        if len(nextCharacter) == 0:
            raise RuntimeError("End of file")

        return nextCharacter

    def getPath(self):
        return self.sourceConfig["path"]



