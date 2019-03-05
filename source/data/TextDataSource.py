
class TextDataSource:
    def __init__(self, config, sourceConfig):
        self.config = config
        self.sourceConfig = sourceConfig
        self.files = self.getFiles()

        self.reset()

    def next(self):
        c = self.file.read(1)

        if len(c) == 0:
            if self.index < len(self.files):
                self.file = open(self.files[self.index], encoding='ISO-8859-1')
                self.index += 1

                c = self.file.read(1)

        return c

    def getPath(self):
        return self.sourceConfig["path"]

    def size(self):
        import os
        return sum([os.path.getsize(f) for f in self.files])

    def reset(self):
        assert len(self.files) > 0
        self.file = open(self.files[0], encoding='ISO-8859-1')
        self.index = 1

    def getName(self):
        return self.getPath()

    def getFiles(self):
        import os
        allFiles = []
        for root, directories, files in os.walk(self.getPath()):
            allFiles += [os.path.join(root, f) for f in files]

        return allFiles




