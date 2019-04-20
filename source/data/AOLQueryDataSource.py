
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
                return elements[1], int(elements[0])
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
        self.file = open(self.files[0], encoding='ISO-8859-1')
        self.index = 1

        self.line = ""
        self.linePosition = 0
        self.random = numpy.random.RandomState(seed=self.getSeed())

    def shuffleDocuments(self):
        self.random.shuffle(self.files)

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
        if not "size" in self.config["adaptor"]["cache"]:
            return 124

        return int(self.config["adaptor"]["cache"]["seed"])



def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

