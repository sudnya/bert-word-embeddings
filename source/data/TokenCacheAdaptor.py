
class TokenCacheAdaptor:
    def __init__(self, config, source):
        self.config = config
        self.source = source
        self.random = numpy.random.RandomState(seed=self.getSeed())

        self.iterations = 0

    def next(self):
        self.refillCache()

        self.iterations += 1

        return self.cache[self.random.randint(len(self.cache))]

    def refillCache():
        if self.iterations < self.getMaximimIterationsPerRefresh():
            return

        self.iterations = 0
        self.cache = []

        while len(self.cache) < self.getCacheSize():
            self.cache.append(self.source.next())

    def getMaximumIterationsPerRefresh(self):
        return int(self.config["adaptor"]["cache"]["reuse"])

    def getCacheSize(self):
        return int(self.config["adaptor"]["cache"]["size"])

    def getSeed(self):
        return int(self.config["adaptor"]["cache"]["seed"])



