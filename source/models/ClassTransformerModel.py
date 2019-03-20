
import logging
import numpy
import os
import json
import shutil
import time
import math
import tensorflow as tf

from models.Vocab import Vocab
from models.ModelDescriptionCheckpointer import ModelDescriptionCheckpointer

logger = logging.getLogger(__name__)

"""Implements a class based transformer model using Tensorflow"""
class ClassTransformerModel:
    def __init__(self, config, trainingDataSource, validationDataSource):
        """Initializes the model.

        Attributes:
            config: The configuration for the model.
            trainingDataSource: list of training samples and labels
            validationDataSource: list of validation samples and labels

        """
        self.config = config
        self.trainingDataSource = trainingDataSource
        self.validationDataSource = validationDataSource
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self.checkpointer = ModelDescriptionCheckpointer(config, self.__class__.__name__)
        self.isLoaded = False

    def train(self):
        """Trains the model.

        Trains the model for epochs specified in the config.
        Runs the validation dataset on the model if specified in the config.
        """
        with self.graph.as_default():
            self.getOrLoadModel()

        for epoch in range(self.getEpochs()):
            self.runOnTrainingDataset(epoch)

            if self.shouldRunValidation():
                self.runOnValidationDataset(epoch)

            self.checkpoint()

    def predict(self, inputs, requestedPredictions):
        with self.graph.as_default():
            self.getOrLoadModel()

        inputs = numpy.array(inputs)

        predictions = self.session.run(self.outputProbabilities,
                feed_dict={self.inputTokens : inputs})

        batchSize = requestedPredictions.shape[0]
        length = requestedPredictions.shape[1]

        outputPredictions = numpy.zeros(requestedPredictions.shape)

        for b in range(batchSize):
            for l in range(length):
                outputPredictions[b,l,:] = \
                    predictions[b,l,requestedPredictions[b,l,:]]
        return outputPredictions

    def getOrLoadModel(self):
        """Returns a linear model.

        If specified, create a new model else load an already existing model.
        """
        self.vocab = Vocab(self.config)

        shouldCreate = not os.path.exists(
            self.checkpointer.getModelDirectory()) or self.getShouldCreateModel()

        if shouldCreate:
            self.createModel()
        else:
            self.loadModel()

    def loadModel(self):
        """Loads an already existing model from the specified path """
        if self.isLoaded:
            return

        self.checkpointer.load()

        directory = self.checkpointer.getModelDirectory()

        logger.debug("Loading checkpoint from: " + str(directory))

        tf.saved_model.loader.load(
            self.session,
            ["serve"],
            directory
        )

        self.setOperationsByName()

        self.isLoaded = True

    def setOperationsByName(self):
        self.inputTokens = self.graph.get_tensor_by_name("input-tokens:0")
        self.labels = self.graph.get_tensor_by_name("output-labels:0")
        self.outputProbabilities = \
            self.graph.get_tensor_by_name("output-probabilities:0")
        self.loss = self.graph.get_tensor_by_name("loss:0")
        self.optimizerStep = self.graph.get_operation_by_name("optimizer-step")

    def createModel(self):
        # inputs (batch, sequence-length)
        self.inputTokens = tf.placeholder(tf.int32, shape=(None, None),
                name="input-tokens")

        # labels (batch, sequence-length)
        self.labels = tf.placeholder(tf.int32, shape=(None, None),
                name="output-labels")

        self.createClassMappings()

        # convert to classes (batch, sequence-length, assignments)
        self.inputClasses = self.convertToClasses(self.inputTokens)
        self.classLabels  = self.convertToClasses(self.labels)

        # class logits (batch, assignmets, sequence-length, class-size)
        classLogits = self.runClassModel(self.inputClasses)

        # convert to vocab logits (batch, sequence-length, vocab-size)
        vocabLogits = self.expandClassLogitsToVocab(classLogits)

        # compute the loss
        self.classLoss = self.evaluateLoss(vocabLogits, self.labels)
        self.vocabLoss = self.evaluateLoss(classLogits, self.classLabels)

        self.loss = self.classLoss + self.vocabLoss

        self.outputProbabilities = tf.nn.softmax(vocabLogits,
                name="output-probabilities")

        # optimizer
        self.optimizerStep = self.createOptimizerStep(self.loss)

        # initializers
        self.globalInitializer = tf.global_variables_initializer()
        self.localInitializer  = tf.local_variables_initializer()

        # summaries
        self.setupSummaries()

        # do the initialization
        self.initializeModel()

    def createClassMappings(self):

        mappings = numpy.zeros([self.getAssignmentCount(), self.vocab.getSize()],
            dtype=numpy.int32)
        weights = numpy.zeros([self.getAssignmentCount(), self.vocab.getSize()],
            dtype=numpy.float32)

        for assignment in range(self.getAssignmentCount()):
            mappings[assignment, :], weights[assignment, :] = self.createMapping(assignment)

        self.classMappings = tf.constant(mappings)
        self.classWeights  = tf.constant(weights)

    def logAdd(self, left, right):

        if left is None:
            return right

        if left == float("-inf"):
            return right
        if right == float("-inf"):
            return left

        return max(left, right) + math.log1p(math.exp( -math.fabs(left - right)))

    def logSumArray(self, array):
        from functools import reduce
        return reduce(lambda x, y : self.logAdd(x, y), array)

    def logSubtract(self, left, right):

        if left <= right:
            assert False, "log of negative number in subtraction " + str(left) + " - " + str(right)

        if right == float("-inf"):
            return left

        return left + math.log1p(-math.exp(right - left))

    def createMapping(self, assignment):
        generator = numpy.random.RandomState(seed=assignment)

        wordCounts = reversed([i * self.getWordFrequencyPowerLawExponent()
            for i in range(self.vocab.getSize())])

        wordCountsPlusRandom = [i + math.log(generator.uniform(0.0, 1.0)) for i in wordCounts]

        logTotalCount = self.logSumArray(wordCountsPlusRandom)

        sortedWordCounts = sorted(enumerate(wordCountsPlusRandom), key=lambda x: x[1], reverse=True)

        logClassSize = logTotalCount - math.log(self.getNumberOfClasses())

        mapping = numpy.zeros([self.vocab.getSize()], dtype=numpy.int32)
        weights = numpy.zeros([self.vocab.getSize()], dtype=numpy.float32)

        currentClass = 0
        wordsInCurrentClass = 0
        logCurrentCount = None
        for wordIndex, logWordCount in sortedWordCounts:
            assert currentClass < self.getNumberOfClasses()
            mapping[wordIndex] = currentClass

            wordsInCurrentClass += 1
            logCurrentCount = self.logAdd(logCurrentCount, logWordCount)
            if logCurrentCount >= logClassSize and currentClass + 1 != self.getNumberOfClasses():
                print(logCurrentCount, logWordCount, currentClass, logClassSize)
                logCurrentCount = self.logSubtract(logCurrentCount, logClassSize)
                wordsInCurrentClass = 0
                currentClass += 1

        currentClass = 0
        currentClassSize = 0
        currentClassMembers = []
        for i, wordCountAndIndex in enumerate(sortedWordCounts):
            wordIndex, wordCount = wordCountAndIndex

            currentClassMembers.append(wordIndex)
            currentClassSize += 1

            # if end of current class
            if ((1 + i) == len(sortedWordCounts) or
                mapping[sortedWordCounts[1 + i][0]] != currentClass):

                for memberIndex in currentClassMembers:
                    weights[memberIndex] = 1.0 / currentClassSize

                print("current class", currentClass, "members", len(currentClassMembers))

                currentClass += 1
                currentClassSize = 0
                currentClassMembers = []

        return mapping, weights

    def initializeModel(self):
        self.session.run(self.globalInitializer)
        self.session.run(self.localInitializer)

    def runOnTrainingDataset(self, epoch):
        """Trains the linear model on the training dataset for one epoch."""
        trainStart = time.time()

        totalLoss = 0.0

        for step in range(self.getStepsPerEpoch()):
            generatorStart = time.time()

            inputs, labels = self.trainingDataSource.next()

            generatorEnd = time.time()

            trainStepStart = time.time()
            loss, gradNorm = self.trainingStep(inputs, labels, step, epoch)
            trainStepEnd = time.time()

            totalLoss += loss

            message = ("Epoch (" + str(epoch) + " / " + str(self.getEpochs()) +
                "), Step (" + str(step) + " / " + str(self.getStepsPerEpoch()) +
                "), Generator time: " + ("%.2f" % (generatorEnd - generatorStart)) +
                ", training step time: " + ("%.2f" % (trainStepEnd -
                    trainStepStart) +
                ", loss: " + str("%.2f" % loss) +
                ", grad norm: " + str("%.2f" % gradNorm)) +
                ", avg-loss: " + str("%.2f" % (totalLoss / (step + 1))))

            print(message, end="\r", flush=True)

        trainEnd = time.time()

        print(message)
        logger.debug(" Training took: " + (str(trainEnd - trainStart)) + " seconds...")

    def trainingStep(self, inputs, labels, step, epoch):
        """Training step for one minibatch of training data."""
        inputs = numpy.array(inputs)
        labels = numpy.array(labels)

        trainingLoss, gradNorm, summaries, _ = self.session.run([self.loss,
            self.gradientNorm, self.mergedSummary, self.optimizerStep],
            feed_dict={self.inputTokens : inputs, self.labels : labels })

        self.trainingSummaryWriter.add_summary(summaries, step + epoch * self.getStepsPerEpoch())
        return trainingLoss, gradNorm

    def runOnValidationDataset(self, epoch):
        """Runs the linear model on the validation dataset for one epoch."""

        validationStart = time.time()

        totalLoss = 0.0

        for step in range(self.getValidationStepsPerEpoch()):
            generatorStart = time.time()

            inputs, labels = self.validationDataSource.next()

            generatorEnd = time.time()

            validationStepStart = time.time()
            loss, summary = self.validationStep(inputs, labels)
            validationStepEnd = time.time()

            totalLoss += loss

            message = ("Validation Step (" + str(step) + " / " +
                    str(self.getValidationStepsPerEpoch()) +
                "), Generator time: " + ("%.2f" % (generatorEnd - generatorStart)) +
                ", validation step time: " + ("%.2f" % (validationStepEnd - validationStepStart)) +
                ", avg-loss: " + ("%.2f" % (totalLoss/(step + 1))))

            print(message, end="\r", flush=True)

        validationEnd = time.time()

        print(message)
        logger.debug(" Validation took: " + (str(validationEnd - validationStart)) + " seconds...")

    def validationStep(self, inputs, labels):
        """One minibatch of validation data processed by the model."""

        inputs = numpy.array(inputs)
        labels = numpy.array(labels)

        validationLoss, summaries = self.session.run([self.loss,
            self.mergedSummary], feed_dict={self.inputTokens : inputs,
                self.labels : labels})
        return validationLoss, summaries

    def createOptimizerStep(self, loss):
        """One step of backprop."""

        optimizer = tf.train.AdamOptimizer(
            learning_rate=float(self.config["model"]["learning-rate"]),
            beta1=0.9,
            beta2=0.999,
            epsilon=numpy.finfo(float).eps,
            name="optimizer-step")

        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients,
        self.config["model"]["gradient-clipping-factor"])
        self.gradientNorm = tf.global_norm(gradients, name="gradient-norm")

        return optimizer.apply_gradients(zip(gradients, variables))

    def setupSummaries(self):
        tf.summary.scalar('cross-entropy', self.loss)
        tf.summary.scalar('gradient-norm', self.gradientNorm)

        self.mergedSummary = tf.summary.merge_all()

        self.trainingSummaryWriter = tf.summary.FileWriter(
            os.path.join(self.getExperimentDirectory(), 'training-summaries'),
            self.graph)

        #if self.shouldRunValidation():
        #    self.validationSummaryWriter = tf.summary.FileWriter(
        #        os.path.join(self.getExperimentDirectory(), 'validation-summaries'),
        #        self.graph)

    def evaluateLoss(self, batchOutputs, labels):
        return tf.identity(tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=batchOutputs
        ), name="loss")

    def convertToClasses(self, inputs):
        # inputs is (batch, sequence)
        # class mappings is (assignments, vocab size)
        # outputs is (batch, sequence, assignments)

        batchSize = tf.shape(inputs)[0]
        sequenceLength = tf.shape(inputs)[1]

        expandedInputs = tf.broadcast_to(tf.reshape(inputs, (batchSize, sequenceLength, 1, 1)),
            (batchSize, sequenceLength, self.getAssignmentCount(), 1))
        expandedClassMappings = tf.broadcast_to(self.classMappings,
            (batchSize, sequenceLength, self.getAssignmentCount(), self.vocab.getSize()))

        # expanded mappings (batch size, sequence legnth, assignment count, vocab size)
        # expanded inputs (batch size, sequence length, assignments, 1)
        classes = tf.batch_gather(expandedClassMappings, expandedInputs)

        return classes

    def expandClassLogitsToVocab(self, classLogits):
        # class logits is (batch size, sequence-length, assignments, class-size)
        # class mappings is (class-assignments, vocab-size)
        # output is (batch-size, sequence-length, vocab-size)

        batchSize      = tf.shape(classLogits)[0]
        sequenceLength = tf.shape(classLogits)[1]

        # class logits is         (batch-size, sequence-length, assignments, class-size)
        # expanded class mappings (batch size, sequence-length, assignments, vocab-size)
        expandedClassMappings = self.broadcastToExpandedDimension(self.classMappings,
            batchSize, sequenceLength)

        gatheredLogits = tf.batch_gather(tf.reshape(classLogits, (-1, self.getNumberOfClasses())),
            tf.reshape(expandedClassMappings, (-1, self.vocab.getSize())))

        gatheredLogits = tf.reshape(gatheredLogits, tf.shape(expandedClassMappings))

        return tf.reduce_sum(tf.multiply(gatheredLogits, self.classWeights), axis=2)

    def broadcastToExpandedDimension(self, tensor, batchSize, sequenceLength):
        classAssignments = tensor.shape[0]
        vocabSize = tensor.shape[1]

        newShape = (batchSize, sequenceLength, classAssignments, vocabSize)

        expandedTensor = tf.broadcast_to(tensor, newShape)

        #print(expandedTensor.shape)

        reshapedTensor = tf.reshape(expandedTensor, newShape)
        #print(reshapedTensor.shape)

        return reshapedTensor

    def runClassModel(self, inputs):
        #print("inputs", inputs.shape)

        # convert sequences to embeddings (output embeddings are Tensor(batch-size, sequence-length, assignments, hidden))
        inputEmbeddings = self.convertToEmbeddings(inputs)

        #print("inputEmbeddings", inputEmbeddings.shape)

        # run encoder (encodedEmbeddings is (batch-size, sequence-length, assignments, hidden))
        encodedEmbeddings = self.runEncoder(inputEmbeddings)

        #print("encodedEmbeddings", encodedEmbeddings.shape)

        # run decoder (logits is Tensor(batch-size, sequence-length, assignments, vocab-size)
        logits = self.runDecoder(encodedEmbeddings)

        #print("logits", logits.shape)

        return logits

    def convertToEmbeddings(self, sequenceIds):
        assignments = []
        for assignment in range(self.getAssignmentCount()):
            assignments.append(self.convertToClassEmbeddings(sequenceIds, assignment))

        return tf.concat(assignments, axis = 2)

    def convertToClassEmbeddings(self, ids, assignment):

        with tf.variable_scope("linear-embeddings", reuse=tf.AUTO_REUSE):
            wordEmbeddingsGlobal = tf.get_variable('class-embeddings-' + str(assignment), \
                    [self.getNumberOfClasses(), self.getEmbeddingSize()])

        wordEmbeddings = tf.nn.embedding_lookup(wordEmbeddingsGlobal, ids[:, :, assignment, :])
        return wordEmbeddings

    def runEncoder(self, embeddings):
        return tf.layers.dense(embeddings, self.getEmbeddingSize(),
        activation="relu")

    def runDecoder(self, inputEmbeddings):
        return tf.layers.dense(inputEmbeddings, self.getNumberOfClasses())

    def checkpoint(self):
        """Creates a checkpoint of current model and saves to model
        directory.
        """

        directory = self.checkpointer.getModelDirectory()
        logger.debug("Saving checkpoint to: " + str(directory))

        self.checkpointer.checkpoint()
        exists = os.path.exists(directory)

        if exists:
            tempDirectory = directory + "-temp"
            shutil.move(directory, tempDirectory)

        with self.graph.as_default():
            tf.saved_model.simple_save(self.session,
                directory,
                inputs={"input-tokens" : self.inputTokens},
                outputs={"output-probabilities" : self.outputProbabilities})

        if exists:
            shutil.rmtree(tempDirectory)


    """Functions to load configuration parameters."""
    def getEmbeddingSize(self):
        return int(self.config["model"]["embedding-size"])

    def getAssignmentCount(self):
        return int(self.config["model"]["assignment-count"])

    def getNumberOfClasses(self):
        return int(self.config["model"]["number-of-classes"])

    def getWordFrequencyPowerLawExponent(self):
        return float(self.config["model"]["word-frequency-power-law-exponent"])

    def shouldRunValidation(self):
        return self.config["model"]["run-validation"]

    def getEpochs(self):
        return int(self.config["model"]["epochs"])

    def getShouldCreateModel(self):
        if not "createNewModel" in self.config["model"]:
            return False
        return bool(self.config["model"]["create-new-model"])

    def getStepsPerEpoch(self):
        return int(self.config["model"]["steps-per-epoch"])

    def getValidationStepsPerEpoch(self):
        return int(self.config["model"]["validation-steps-per-epoch"])

    def getExperimentDirectory(self):
        return self.config["model"]["directory"]











