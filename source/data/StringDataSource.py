class StringDataSource:
    def __init__(self, theString):
        self.string = theString

        self.reset()

    def next(self):
        if self.index >= len(self.string):
            return ''

        c = self.string[self.index]

        self.index += 1

        return c

    def size(self):
        return len(self.string)

    def reset(self):
        self.index = 0


