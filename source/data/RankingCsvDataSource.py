
import numpy
import csv

class RankingCsvDataSource:
    def __init__(self, config, sourceConfig):
        self.config = config
        self.sourceConfig = sourceConfig
        self.files = self.getFiles()

        self.reset()

    def next(self):
        self.fillLineBuffer()

        if len(self.line) == 0:
            return "", -1

        nextCharacter = self.line[self.linePosition]

        self.linePosition += 1

        return nextCharacter, self.lineId

    def fillLineBuffer(self):
        if self.linePosition < len(self.line):
            return

        self.linePosition = 0
        self.line, self.lineId = self.parseLine()

        while len(self.line) == 0:
            self.index += 1

            if self.index >= len(self.files):
                break

            self.reader = csv.reader(open(self.files[self.getIndex()], encoding='ISO-8859-1'),
                delimiter=',', quotechar='"')

            self.line, self.lineId = self.parseLine()

    def parseLine(self):
        nextLine = self.readline()

        while not nextLine is None:
            try:
                message = nextLine[0]
                if len(message) == 0:
                    nextLine = self.readline()
                    continue
                return message, 1 if float(nextLine[1]) < 1200 else 0
            except Exception as e:
                nextLine = self.readline()

        return "", -1

    def readline(self):
        try:
            return next(self.reader)
        except:
            return None

    def getPath(self):
        return self.sourceConfig["path"]

    def reset(self):
        assert len(self.files) > 0, "No files found in " + self.getPath()
        self.indices = list(range(len(self.files)))
        self.index = 0

        self.reader = csv.reader(open(self.files[0], encoding='ISO-8859-1'),
            delimiter=',', quotechar='"')

        self.line = ""
        self.linePosition = 0

        self.random = numpy.random.RandomState(seed=self.getSeed())

    def shuffleDocuments(self):
        self.random.shuffle(self.indices)
        self.index = 0
        self.file = csv.reader(open(self.files[self.getIndex()], encoding='ISO-8859-1'),
            delimiter=',', quotechar='"')

    def getIndex(self):
        return self.indices[self.index]

    def clone(self):
        return RankingCsvDataSource(self.config, self.sourceConfig)

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




