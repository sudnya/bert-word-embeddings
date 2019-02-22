class LinearModel:
    def __init__(self, config):
        self.config = config
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        # create a model - all vars / weights / global & local 
        


    def train(self):
        with self.graph.as_default():
            self.getOrLoadModel()
            
        for epoch in range(self.getEpochs()):
            self.runOnTrainingDataset(epoch)

            if self.shouldRunValidation():
                self.runOnValidationDataset(epoch)

            self.checkpoint()


    def getOrLoadModel(self):
        shouldCreate = not os.path.exists(self.getCheckpointJSONFilePath()) or self.shouldCreateModel()

        with self.graph.as_default():
            if shouldCreate:
                return self.createModel()
            else:
                return self.loadModel()

    def createModel(self):
        # inputs

        ## (batch, walks, walk-length, embedding-size)
        self.embeddings = tf.placeholder(tf.float32,
            shape=(None, None, None, self.getTotalEmbeddingSize()),
            name = "node-embeddings")

        self.loss = self.evaluateLoss()

        # optimizer
        self.optimizerStep = self.createOptimizerStep()

        # initializers
        self.globalInitializer = tf.global_variables_initializer()
        self.localInitializer  = tf.local_variables_initializer()

        # summaries
        self.setupSummaries()

        # do the initialization
        self.initializeModel()

    def loadModel(self):
        jsonPath = self.getCheckpointJSONFilePath()

        with open(jsonPath) as jsonFile:
            configuration = json.load(jsonFile)

        assert configuration["type"] == "LinearModel"

        checkpointDirectoryPath = os.path.join(os.path.dirname(jsonPath), "checkpoint")

        logger.debug("Loading checkpoint from: " + str(checkpointDirectoryPath))
        tf.saved_model.loader.load(
            self.session,
            ["serve"],
            checkpointDirectoryPath
        )

        self.version = configuration["version"]

        self.setOperationsByName()

        self.isLoaded = True

        

    def runOnTrainingDataset(self, epoch):
        import time
        trainStart = time.time()

        for step in range(self.getStepsPerEpoch()):
            generatorStart = time.time()

            inputs, labels = self.trainingData.next()

            generatorEnd = time.time()

            trainStepStart = time.time()
            self.trainingStep(inputs, labels)
            trainStepEnd = time.time()

            message = ("Epoch (" + str(epoch) + " / " + str(self.getEpochs()) +
                "), Step (" + str(step) + " / " + str(self.getStepsPerEpoch()) +
                "), Generator time: " + ("%.2f" % (generatorEnd - generatorStart)) +
                ", training step time: " + ("%.2f" % (trainStepEnd - trainStepStart)))

            print(message, end="\r", flush=True)

        trainEnd = time.time()

        print(message)
        logger.debug(" Training took: " + (str(trainEnd - trainStart)) + " seconds...")

    def trainingStep(self, inputs, labels):
        pass
        # call a function to evaluate the above variables
        

    def evaluateLoss(self):
        pass


    def processInputMiniBatch(self, inputs, ):
        while True:
            runEncoderDecoder()
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


