
import numpy

class AOLQueryDataSource:
    def __init__(self, config, sourceConfig):
        self.config = config
        self.sourceConfig = sourceConfig
        self.files = self.getFiles()

        self.reset()

    def next(self):
        self.fillLineBuffer()

        if len(self.line) == 0:
            return ""

        nextCharacter = self.line[self.linePosition]

        self.linePosition += 1

        return nextCharacter, self.userId

    def fillLineBuffer(self):
        if self.linePosition < len(self.line):
            return

        self.linePosition = 0
        self.line, self.userId = self.parseLine()

        while len(self.line) == 0:
            if self.index >= len(self.files):
                break

            self.file = open(self.files[self.index], encoding='ISO-8859-1')
            self.index += 1

            self.line, self.userId = self.parseLine()

    def parseLine(self):
        nextLine = self.readline()

        while len(nextLine) > 0:
            elements = nextLine.split('\t')
            if isInt(elements[0]):
                return elements[1] + "\n", int(elements[0]) % self.getMaximumId()
            nextLine = self.readline()

        return "", -1

    def readline(self):
        return self.file.readline()

    def getPath(self):
        return self.sourceConfig["path"]

    def size(self):
        assert False, "not implemented"

    def reset(self):
        assert len(self.files) > 0, "No files found in " + self.getPath()
        self.indices = list(range(len(self.files)))
        self.index = 0
        self.random = numpy.random.RandomState(seed=self.getSeed())

        self.file = open(self.files[self.getIndex()], encoding='ISO-8859-1')

        self.line = ""
        self.linePosition = 0

    def shuffleDocuments(self):
        self.random.shuffle(self.indices)
        self.index = 0
        self.file = open(self.files[self.getIndex()], encoding='ISO-8859-1')

        self.line = ""
        self.linePosition = 0

    def clone(self):
        return AOLQueryDataSource(self.config, self.sourceConfig)

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
        if not "seed" in self.sourceConfig:
            return 124

        return int(self.sourceConfig["seed"])

    def getMaximumId(self):
        if not "maximum-document-id" in self.sourceConfig:
            return 1024

        return int(self.sourceConfig["maximum-document-id"])

    def getIndex(self):
        return self.indices[self.index]

def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


