
class TextDataSource:
    def __init__(self, config, sourceConfig):
        self.config = config
        self.sourceConfig = sourceConfig

        self.reset()

    def next(self):
        return self.file.read(1)

    def getPath(self):
        return self.sourceConfig["path"]

    def size(self):
        import os
        return os.path.getsize(self.getPath())

    def reset(self):
        self.file = open(self.getPath(), encoding='ISO-8859-1')




