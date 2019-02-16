

class LinearModel:
    def __init__(self, config):
        self.config = config


    def train(self):
        pass # NOT implemented

    def runEncoderDecoder(self, inputSequence, historySequence):
        #   inputSequence Tensor(batch-size, sequence-length) of ints
        # historySequence Tensor(batch-size, sequence-length - 1) of ints

        # convert sequences to embeddings (output embeddings are Tensor(batch-size, sequence-length, hidden))
        inputEmbeddings = self.convertToEmbeddings(inputSequence)
        historyEmbeddings = self.convertToEmbeddings(historySequence)

        # run encoder (encodedEmbeddings is (batch-size, sequence-length, hidden))
        encodedEmbeddings = self.runEncoder(inputEmbeddings)

        # run decoder (decoded embeddings is Tensor(batch-size, hidden))
        decodedEmbeddings = self.runDecoder(encodedEmbeddings, historyEmbeddings)

        # run softmax (probabilities is Tensor(batch-size, vocab-size)
        probabilities = self.runSoftmax(decodedEmbeddings)

        return probabilities


