
import numpy

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

        return c, self.index

    def getPath(self):
        return self.sourceConfig["path"]

    def size(self):
        import os
        return sum([os.path.getsize(f) for f in self.files])

    def reset(self):
        assert len(self.files) > 0, "No files found in " + self.getPath()
        self.file = open(self.files[0], encoding='ISO-8859-1')
        self.index = 1
        self.random = numpy.random.RandomState(seed=self.getSeed())

    def shuffleDocuments(self):
        self.random.shuffle(self.files)
        self.file = open(self.files[0], encoding='ISO-8859-1')
        self.index = 1

    def clone(self):
        return TextDataSource(self.config, self.sourceConfig)

    def getName(self):
        return self.getPath()

    def getFiles(self):
        import os

        if os.path.isfile(self.getPath()):
            return [self.getPath()]

        allFiles = []

        for root, directories, files in os.walk(self.getPath()):
            allFiles += [os.path.join(root, f) for f in files]

        return sorted(allFiles)

    def getSeed(self):
        if not "size" in self.config["adaptor"]["cache"]:
            return 125

        return int(self.config["adaptor"]["cache"]["seed"])




