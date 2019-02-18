class LinearModel:
    def __init__(self, config):
        self.config = config


    def train(self):
        pass # NOT implemented

    def runEncoderDecoder(self, inputSequence, historicSequence):
        # inputSequence Tensor(batch-size, sequence-length) of ints
        # historySequence Tensor(batch-size, sequence-length - 1) of ints

        # convert sequences to embeddings (output embeddings are Tensor(batch-size, sequence-length, hidden))
        inputEmbeddings   = self.convertToEmbeddings(inputSequence)
        historyEmbeddings = self.convertToEmbeddings(historicSequence)

        # run encoder (encodedEmbeddings is (batch-size, sequence-length, hidden))
        encodedEmbeddings = self.runEncoder(inputEmbeddings)

        # run decoder (decoded embeddings is Tensor(batch-size, hidden))
        decodedEmbeddings = self.runDecoder(encodedEmbeddings, historicEmbeddings)

        # run softmax (probabilities is Tensor(batch-size, vocab-size)
        probabilities = self.runSoftmax(decodedEmbeddings)

        return probabilities


    def convertToEmbeddings(self, sequence_ids):
        word_embeddings = tf.get_variable('word_embeddings', \
                [config.get('vocabulary_size'), config.get('embedding_size')])
        embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, sequence_ids)
        return embedded_word_ids

    def runEncoder(self, embeddings):
        return tf.layers.dense(embeddings, config.get('hidden_encoding'))

    def runDecoder(self, inputEmbeddings, historicEmbeddings):
        return tf.layers.dense(tf.concat(inputEmbeddings, historicEmbeddings), \
                config.get('embedding_size'))

    def runSoftmax(self, decodedEmbeddings):
        return tf.nn.softmax(decodedEmbeddings)


