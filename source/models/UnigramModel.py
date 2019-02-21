
from models.Vocab import Vocab

logger = logging.getLogger(__name__)

class UnigramModel:
    def __init__(self, config, trainingData, validationData):
        self.config = config
        self.trainingData = trainingData
        self.validationData = validationData
        self.checkpointer = ModelDescriptionCheckpointer(config, "UnigramModel")

        self.getOrLoadModel()

    def train(self):
        for epoch in range(self.getEpochs()):
            self.runOnTrainingDataset(epoch)

            if self.shouldRunValidation():
                self.runOnValidationDataset(epoch)

            self.checkpoint()

    def runOnTrainingDataset(epoch):
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

        logger.debug(message)
        logger.debug(" Training took: " + (str(trainEnd - trainStart)) + " seconds...")

    def trainingStep(self, inputs, labels):
        # just consider the labels
        for batch in range(labels.shape[0]):
            self.totalTokens += labels.shape[1]

            for token in range(labels.shape[1]):
                if not token in self.tokenCounts:
                    self.tokenCounts[token] = 0

                self.tokenCounts[token] += 1

    def runOnValidationDataset(epoch):
        start = time.time()

        for step in range(self.getValidationStepsPerEpoch()):
            generatorStart = time.time()

            inputs, labels = self.validationData.next()

            generatorEnd = time.time()

            stepStart = time.time()
            crossEntropy, tokens = self.validationStep(inputs, labels)
            stepEnd = time.time()

            message = ("Epoch (" + str(epoch) + " / " + str(self.getEpochs()) +
                "), Step (" + str(step) + " / " + str(self.getValidationStepsPerEpoch()) +
                "), Generator time: " + ("%.2f" % (generatorEnd - generatorStart)) +
                ", validation step time: " + ("%.2f" % (stepEnd - stepStart)))

            print(message, end="\r", flush=True)

        end = time.time()

        logger.debug(message)
        logger.debug(" Validation took: " + (str(end - start)) + " seconds...")

    def validationStep(self, inputs, labels):
        crossEntropy = 0.0
        for batch in range(labels.shape[0]):
            for token in range(labels.shape[1]):
                tokenProbability = self.getTokenProbability(token)
                crossEntropy += -math.log(tokenProbability)

        return crossEntropy, labels.shape[0] * labels.shape[1]

    def getOrloadModel(self):
        self.vocab = Vocab(self.config)

        shouldCreate = not os.path.exists(
            self.checkpointer.getModelDirectory()) or self.shouldCreateModel()

        if shouldCreate:
            return self.createModel()
        else:
            return self.load()

    def createModel(self):
        self.tokenCounts = numpy.zeros(self.getVocabSize())
        self.totalTokens = 0

    def checkpoint(self):

        directory = self.checkpointer.getModelDirectory()
        logger.debug("Saving checkpoint to: " + str(directory))

        self.checkpointer.checkpoint()

        exists = os.path.exists(directory)
        if exists:
            tempDirectory = directory + "-temp"

            shutil.move(directory, tempDirectory)

        os.mkdirs(directory)
        with open(os.path.join(directory, "unigram-statistics.json"), "w") as jsonFile:
            json.dump([self.totalTokens, self.tokenCounts], jsonFile)

        if exists:
            shutil.rmtree(tempDirectory)


    def load(self):
        self.checkpointer.load()

        directory = self.checkpointer.getModelDirectory()

        logger.debug("Loading checkpoint from: " + str(directory))


