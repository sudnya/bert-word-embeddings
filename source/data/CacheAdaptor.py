
import numpy

import logging

logger = logging.getLogger(__name__)

class CacheAdaptor:
    def __init__(self, config, source):
        self.config = config
        self.source = source
        self.random = numpy.random.RandomState(seed=self.getSeed())

        self.iterations = None

    def next(self):
        self.refillCache()

        self.iterations += 1

        return self.cache[self.random.randint(len(self.cache))]

    def refillCache(self):
        if not self.iterations is None and self.iterations < self.getMaximumIterationsPerRefresh():
            return

        self.iterations = 0
        self.cache = []

        logger.debug("Filling cache with " + str(self.getCacheSize()) + " samples...")

        while len(self.cache) < self.getCacheSize():
            self.cache.append(self.source.next())

    def getMaximumIterationsPerRefresh(self):
        return self.getReuse() * self.getCacheSize()

    def getReuse(self):
        if not "reuse" in self.config["adaptor"]["cache"]:
            return 1

        return int(self.config["adaptor"]["cache"]["reuse"])

    def getCacheSize(self):
        if not "size" in self.config["adaptor"]["cache"]:
            return 32

        return int(self.config["adaptor"]["cache"]["size"])

    def getSeed(self):
        if not "size" in self.config["adaptor"]["cache"]:
            return 121

        return int(self.config["adaptor"]["cache"]["seed"])



