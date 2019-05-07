
class PassThroughLabelAdaptor:
    def __init__(self, config, source):
        self.config = config
        self.source = source

    def next(self):
        chunk = self.source.next()

        chunk, documentId = zip(*chunk)

        return chunk, chunk, chunk, chunk

    def reset(self):
        self.source.reset()

    def size(self):
        return self.source.size()

    def setMaximumSize(self, size):
        self.source.setMaximumSize(size)



