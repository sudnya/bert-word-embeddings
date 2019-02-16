
class TextDataSource:
    def __init__(self, config, sourceConfig):
        self.config = config
        self.sourceConfig = sourceConfig
        self.file = open(self.getPath(), encoding='utf-8')

    def next(self):
        return self.file.read(1)

    def getPath(self):
        return self.sourceConfig["path"]



