

import numpy

class RedditDataSource:
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

        return nextCharacter

    def fillLineBuffer(self):
        if self.linePosition < len(self.line):
            return

        self.linePosition = 0
        self.line = self.parseLine()

        while len(self.line) == 0:
            if self.index >= len(self.files):
                break

            self.file = open(self.files[self.index], encoding='ISO-8859-1')
            self.index += 1

            self.line = self.parseLine()

    def parseLine(self):
        nextLine = self.readline()

        while len(nextLine) > 0:
            try:
                message = json.loads(nextLine)
                return message["body"], message["subreddit"]
            except:
                nextLine = self.readline()

        return "", -1

    def readline(self):
        return self.file.readline()

    def getPath(self):
        return self.sourceConfig["path"]

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
        return RedditDataSource(self.config, self.sourceConfig)

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
            return 126

        return int(self.config["adaptor"]["cache"]["seed"])




