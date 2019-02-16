

from data.AdaptorFactory import AdaptorFactory

class DataSources:
    def __init__(self, config):
        self.currentSource = 0
        self.sources = []
        self.config = config

    def next(self):
        item = self.sources[self.currentSource].next()

        self.currentSource = (self.currentSource + 1) % len(self.sources)

        return item

    def addSource(self, source):
        adaptor = AdaptorFactory(self.config).create(source)

        self.sources.append(adaptor)




