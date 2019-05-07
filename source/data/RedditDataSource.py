

import numpy
import json

class RedditDataSource:
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

        return nextCharacter, self.subredditId

    def fillLineBuffer(self):
        if self.linePosition < len(self.line):
            return

        self.linePosition = 0
        self.line, self.subredditId = self.parseLine()

        while len(self.line) == 0:
            self.index += 1

            if self.index >= len(self.files):
                break

            self.file = open(self.files[self.getIndex()], encoding='ISO-8859-1')

            self.line, self.subredditId = self.parseLine()

    def parseLine(self):
        nextLine = self.readline()

        while len(nextLine) > 0:
            try:
                message = json.loads(nextLine)
                if len(message["body"]) == 0:
                    nextLine = self.readline()
                    continue
                return message["body"], self.getSubRedditId(message["subreddit"])
            except Exception as e:
                nextLine = self.readline()

        return "", -1

    def getSubRedditId(self, subreddit):
        if not subreddit in self.subreddits:
            self.subreddits[subreddit] = len(self.subreddits)

        return self.subreddits[subreddit] % self.getMaximumId()

    def readline(self):
        return self.file.readline()

    def getPath(self):
        return self.sourceConfig["path"]

    def reset(self):
        assert len(self.files) > 0, "No files found in " + self.getPath()
        self.indices = list(range(len(self.files)))
        self.index = 0

        self.file = open(self.files[0], encoding='ISO-8859-1')

        self.subreddits = {}

        self.line = ""
        self.linePosition = 0

        self.random = numpy.random.RandomState(seed=self.getSeed())

    def shuffleDocuments(self):
        self.random.shuffle(self.indices)
        self.index = 0
        self.file = open(self.files[self.getIndex()], encoding='ISO-8859-1')

    def getIndex(self):
        return self.indices[self.index]

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
        if not "seed" in self.sourceConfig:
            return 124

        return int(self.sourceConfig["seed"])

    def getMaximumId(self):
        if not "maximum-document-id" in self.sourceConfig:
            return 1024

        return int(self.sourceConfig["maximum-document-id"])




