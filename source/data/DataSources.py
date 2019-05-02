

from data.AdaptorFactory import AdaptorFactory

from profilehooks import profile

class DataSources:
    def __init__(self, config):
        self.currentSource = 0
        self.sources = []
        self.config = config

#    @profile
    def next(self):
        item = None

        for i in range(len(self.sources)):
            try:
                item = self.sources[self.currentSource].next()
                self.currentSource = (self.currentSource + 1) % len(self.sources)
                break
            except:
                self.currentSource = (self.currentSource + 1) % len(self.sources)

        if item is None:
            raise StopIteration()

        return item

    def addSource(self, source):
        adaptor = AdaptorFactory(self.config).create(source)

        self.sources.append(adaptor)

    def reset(self):
        for source in self.sources:
            source.reset()

    def size(self):
        return sum([source.size() for source in self.sources])

    def setMaximumSize(self, size):
        for source in self.sources:
            source.setMaximumSize(size//len(self.sources))




