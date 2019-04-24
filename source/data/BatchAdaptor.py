
import numpy

class BatchAdaptor:
    def __init__(self, config, source):
        self.config = config
        self.source = source

    def next(self):
        samples = [self.source.next() for i in range(self.getBatchSize())]

        return (numpy.array([sample[0] for sample in samples]),
                numpy.array([sample[1] for sample in samples]),
                numpy.array([sample[2] for sample in samples]),
                numpy.array([sample[3] for sample in samples]))

    def getBatchSize(self):
        if not "size" in self.config["adaptor"]["batching"]:
            return 1

        return int(self.config["adaptor"]["batching"]["size"])


    def size(self):
        return self.source.size() // self.getBatchSize()

    def reset(self):
        return self.source.reset()

    def setMaximumSize(self, size):
        self.source.setMaximumSize(size * self.getBatchSize())



