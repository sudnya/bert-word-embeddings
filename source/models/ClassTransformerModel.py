
import logging
import numpy
import os
import json
import shutil
import time
import math
import humanize
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
        self.bestValidationLoss = None

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
                self.validationDataSource.reset()

            self.checkpointBestModel()
            self.trainingDataSource.reset()

    def checkpointBestModel(self):
        if self.bestValidationLoss is None:
            self.checkpoint("best")
            return

        if self.totalLoss < self.bestValidationLoss:
            logger.info("Updating best model with loss: " + str(self.totalLoss))
            self.bestValidationLoss = self.totalLoss
            self.checkpoint("best")
        else:
            self.checkpoint("checkpoint")


    def predict(self, inputs, requestedPredictions):
        with self.graph.as_default():
            self.getOrLoadModel()

        assert False, "Not Implemented"

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

    def getFeatures(self, inputs, secondInputs):
        with self.graph.as_default():
            self.getOrLoadModel()

        inputs = numpy.expand_dims(numpy.array(inputs), axis=2)
        secondInputs = numpy.expand_dims(numpy.array(secondInputs), axis=2)
        inputs = numpy.concatenate([inputs, secondInputs], axis=2)

        # (batch, sequence, 2, embedding-size)
        predictions = self.session.run(self.features,
                feed_dict={self.inputTokens : inputs})

        return predictions[:, :, 0, :]


    def getOrLoadModel(self):
        """Returns a linear model.

        If specified, create a new model else load an already existing model.
        """
        if self.isLoaded:
            return

        self.vocab = Vocab(self.config)

        shouldCreate = not os.path.exists(
            self.checkpointer.getModelLoadDirectory()) or self.getShouldCreateModel()

        if shouldCreate:
            self.createModel()
        else:
            self.loadModel()

        self.logModel()

    def logModel(self):
        totalParameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variableParameters = 1
            for dim in shape:
                variableParameters *= dim.value
            totalParameters += variableParameters
            logger.debug("Variable '" + variable.name + "' " +
                str(humanize.naturalsize(variableParameters)) + " (params) " +
                str(shape) + " (dims)")

        logger.debug("Total #params '" + str(humanize.naturalsize(totalParameters)) + "' ")

    def loadModel(self):
        """Loads an already existing model from the specified path """

        self.checkpointer.load()

        directory = self.checkpointer.getModelLoadDirectory()

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
        self.features = self.graph.get_tensor_by_name("features:0")
        self.vocabLoss = self.graph.get_tensor_by_name("vocab-loss:0")
        self.classificationLoss = self.graph.get_tensor_by_name("classification-loss:0")
        self.classLoss = self.graph.get_tensor_by_name("class-loss:0")
        self.outputProbabilities = self.graph.get_tensor_by_name("output-probabilities:0")
        self.loss = self.graph.get_tensor_by_name("loss:0")
        self.optimizerStep = self.graph.get_operation_by_name("optimizer-step")

    def createModel(self):
        # inputs (batch, sequence-length, 2)
        self.inputTokens = tf.placeholder(tf.int32, shape=(None, None, 2),
                name="input-tokens")

        # labels (batch, sequence-length, 2)
        self.labels = tf.placeholder(tf.int32, shape=(None, None, 2),
                name="output-labels")

        self.createClassMappings()

        # convert to classes (batch, sequence-length, 2, assignments)
        self.inputClasses = self.convertToClasses(self.inputTokens)
        self.classLabels  = self.convertToClasses(self.labels)

        # class logits (batch, sequence-length, 2, assignmets, class-size)
        classLogits = self.runClassModel(self.inputClasses)

        # classification logits (batch, sequence-length, 2, assignments, 2)
        classificationLogits = self.runClassificationModel()

        # compute the losses
        self.clusterLoss = tf.identity(self.evaluateClusteringLoss(
            self.features, self.classLabels), name="clustering-loss")
        self.classificationLoss = tf.identity(self.evaluateClassificationLoss(
            classificationLogits, self.classLabels), name="classification-loss")
        self.classLoss = tf.identity(self.evaluateLoss(classLogits[:, 1:, :, :, :],
            self.classLabels[:, 1:, :, :]), name="class-loss")
        self.vocabLoss = tf.identity(self.evaluateVocabLoss(classLogits[:, 1:, :, :, :],
            self.labels[:, 1:, :]), name="vocab-loss")

        self.loss = tf.identity(self.classLoss +
            self.classificationLoss +
            self.clusterLoss +
            self.vocabLoss,
            name="loss")

        # convert to vocab logits (batch, sequence-length, vocab-size)
        vocabLogits = self.expandClassLogitsToVocab(classLogits)

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

        self.classMappingsHost = mappings
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

        assert self.getNumberOfDirectClasses() <= self.getNumberOfClasses()
        assert self.getNumberOfDirectClasses() <= self.vocab.getSize()

        vocabSize       = self.vocab.getSize() - self.getNumberOfDirectClasses()
        numberOfClasses = self.getNumberOfClasses() - self.getNumberOfDirectClasses()

        directMapping = numpy.arange(self.getNumberOfDirectClasses(), dtype=numpy.int32)
        directWeights = numpy.ones(self.getNumberOfDirectClasses(), dtype=numpy.float32)

        mapping, weights = self.createLogMapping(assignment, vocabSize, numberOfClasses)

        return (numpy.concatenate([directMapping, self.getNumberOfDirectClasses() + mapping]),
            numpy.concatenate([directWeights, weights]))

    def createLogMapping(self, assignment, vocabSize, numberOfClasses):

        generator = numpy.random.RandomState(seed=assignment)

        wordCounts = reversed([i * self.getWordFrequencyPowerLawExponent()
            for i in range(vocabSize)])

        wordCountsPlusRandom = [i + math.log(generator.uniform(0.0, 1000.0)) for i in wordCounts]

        logTotalCount = self.logSumArray(wordCountsPlusRandom)

        sortedWordCounts = sorted(enumerate(wordCountsPlusRandom), key=lambda x: x[1], reverse=True)

        logClassSize = logTotalCount - math.log(numberOfClasses)

        mapping = numpy.zeros([vocabSize], dtype=numpy.int32)
        weights = numpy.zeros([vocabSize], dtype=numpy.float32)

        currentClass = 0
        wordsInCurrentClass = 0
        logCurrentCount = None
        for wordIndex, logWordCount in sortedWordCounts:
            assert currentClass < numberOfClasses
            mapping[wordIndex] = currentClass

            wordsInCurrentClass += 1
            logCurrentCount = self.logAdd(logCurrentCount, logWordCount)
            if logCurrentCount >= logClassSize and currentClass + 1 != numberOfClasses:
                #print(logCurrentCount, logWordCount, currentClass, logClassSize)
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

                if currentClass == 0 or i == (len(sortedWordCounts) - 1):
                    logger.info("current class " + str(currentClass) +
                        " members " + str(len(currentClassMembers)))

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

            try:
                inputs, labels, secondInputs, secondLabels = self.trainingDataSource.next()
            except:
                break

            generatorEnd = time.time()

            trainStepStart = time.time()
            loss, gradNorm = self.trainingStep(inputs, labels, secondInputs, secondLabels,
                                               step, epoch)
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

    def trainingStep(self, inputs, labels, secondInputs, secondLabels, step, epoch):
        """Training step for one minibatch of training data."""
        inputs = numpy.expand_dims(numpy.array(inputs), axis=2)
        labels = numpy.expand_dims(numpy.array(labels), axis=2)
        secondInputs = numpy.expand_dims(numpy.array(secondInputs), axis=2)
        secondLabels = numpy.expand_dims(numpy.array(secondLabels), axis=2)

        inputs = numpy.concatenate([inputs, secondInputs], axis=2)
        labels = numpy.concatenate([labels, secondLabels], axis=2)

        trainingLoss, gradNorm, summaries, _ = self.session.run([self.loss,
            self.gradientNorm, self.mergedSummary, self.optimizerStep],
            feed_dict={self.inputTokens : inputs, self.labels : labels })

        if step % self.getStepsPerTensorboardLog():
            self.trainingSummaryWriter.add_summary(summaries,
                step + epoch * self.getStepsPerEpoch())
        return trainingLoss, gradNorm

    def runOnValidationDataset(self, epoch):
        """Runs the linear model on the validation dataset for one epoch."""

        validationStart = time.time()

        self.totalLoss = 0.0
        self.totalVocabLoss = 0.0

        for step in range(self.getValidationStepsPerEpoch()):
            generatorStart = time.time()

            inputs, labels, secondInputs, secondLabels = self.validationDataSource.next()

            generatorEnd = time.time()

            validationStepStart = time.time()
            loss, vocabLoss = self.validationStep(inputs, labels, secondInputs, secondLabels)
            validationStepEnd = time.time()

            self.totalLoss += loss
            self.totalVocabLoss += vocabLoss

            message = ("Validation Step (" + str(step) + " / " +
                    str(self.getValidationStepsPerEpoch()) +
                "), Generator time: " + ("%.2f" % (generatorEnd - generatorStart)) +
                ", validation step time: " + ("%.2f" % (validationStepEnd - validationStepStart)) +
                ", avg-loss: " + ("%.2f" % (self.totalLoss/(step + 1))))

            print(message, end="\r", flush=True)

        validationEnd = time.time()

        print(message)
        logger.debug(" Validation took: " + (str(validationEnd - validationStart)) + " seconds...")

        self.addValidationSummaries(self.totalLoss, self.totalVocabLoss, epoch)

    def addValidationSummaries(self, totalLoss, vocabLoss, epoch):

        averageLoss = totalLoss / self.getValidationStepsPerEpoch()

        summary = tf.Summary(value=[
            tf.Summary.Value(tag="validation-loss", simple_value=averageLoss),
        ])

        self.trainingSummaryWriter.add_summary(summary, epoch)

        averageVocabLoss = vocabLoss / self.getValidationStepsPerEpoch()

        summary = tf.Summary(value=[
            tf.Summary.Value(tag="validation-vocab-cross-entropy", simple_value=averageVocabLoss),
        ])

        self.trainingSummaryWriter.add_summary(summary, epoch)

    def validationStep(self, inputs, labels, secondInputs, secondLabels):
        """One minibatch of validation data processed by the model."""

        inputs = numpy.expand_dims(numpy.array(inputs), axis=2)
        labels = numpy.expand_dims(numpy.array(labels), axis=2)
        secondInputs = numpy.expand_dims(numpy.array(secondInputs), axis=2)
        secondLabels = numpy.expand_dims(numpy.array(secondLabels), axis=2)

        inputs = numpy.concatenate([inputs, secondInputs], axis=2)
        labels = numpy.concatenate([labels, secondLabels], axis=2)

        validationLoss, vocabLoss = self.session.run([self.loss, self.vocabLoss],
                feed_dict={self.inputTokens : inputs,
                self.labels : labels})
        return validationLoss, vocabLoss

    def createOptimizerStep(self, loss):
        """One step of backprop."""

        optimizer = tf.train.AdamOptimizer(
            learning_rate=float(self.config["model"]["learning-rate"]),
            beta1=0.9,
            beta2=0.98,
            epsilon=10e-9,
            name="optimizer-step")

        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients,
        self.config["model"]["gradient-clipping-factor"])
        self.gradientNorm = tf.global_norm(gradients, name="gradient-norm")

        return optimizer.apply_gradients(zip(gradients, variables))

    def setupSummaries(self):
        tf.summary.scalar('total-loss', self.loss)
        tf.summary.scalar('document-class-cross-entropy', self.classificationLoss)
        tf.summary.scalar('vocab-cross-entropy', self.vocabLoss)
        tf.summary.scalar('class-cross-entropy', self.classLoss)
        tf.summary.scalar('cluster-loss', self.clusterLoss)
        tf.summary.scalar('gradient-norm', self.gradientNorm)

        self.mergedSummary = tf.summary.merge_all()

        self.trainingSummaryWriter = tf.summary.FileWriter(
            os.path.join(self.getExperimentDirectory(), 'training-summaries'),
            self.graph)

        #if self.shouldRunValidation():
        #    self.validationSummaryWriter = tf.summary.FileWriter(
        #        os.path.join(self.getExperimentDirectory(), 'validation-summaries'),
        #        self.graph)

    def evaluateClusteringLoss(self, features, classLabels):
        # features is [batch, sequence, 2, assignments, feature-dimension]
        # class labels is [batch, sequence, 2, assignments]
        assignmentLosses = []

        batchSize = tf.shape(features)[0]
        sequenceLength = tf.shape(features)[1]

        features = tf.reshape(self.features, (batchSize, sequenceLength,
            2, self.getAssignmentCount(), self.getEmbeddingSize()))

        for i in range(self.getAssignmentCount()):
            assignmentLosses.append(self.evaluatePerAssignmentClusterLoss(
                features[:, :, :, i, :], classLabels[:, :, :, i]))

        return sum(assignmentLosses) / (tf.multiply(tf.cast(batchSize, dtype=tf.float32),
            2.0 * self.getAssignmentCount()))

    def evaluatePerAssignmentClusterLoss(self, features, labels):
        # features is [batch, sequence, 2, feature-dim]
        # labels is [batch, sequence, 2]
        wordFeatures = tf.reshape(features[:, 0, :, :], (-1, self.getEmbeddingSize()))
        tripletLabels = tf.reshape(labels[:, 0, :], (-1, ))

        return self.tripletLoss(wordFeatures, tripletLabels)

    def tripletLoss(self, features, labels):
        return tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, features)

    def evaluateClassificationLoss(self, batchOutputs, labels):
        # batch outputs is [batch, assignments, 2]
        # labels is [batch, sequence, assignments, 2]
        labels = tf.cast(labels[:, 0, :, 0] == labels[:, 0, :, 1], tf.int32)
        return tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=batchOutputs)

    def evaluateLoss(self, batchOutputs, labels):
        return tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=batchOutputs)

    def klDivergence(self, a, b):
        a = tf.distributions.Categorical(probs=a + numpy.finfo(float).eps)
        b = tf.distributions.Categorical(probs=tf.nn.softmax(b) + numpy.finfo(float).eps)
        return tf.reduce_mean(tf.distributions.kl_divergence(a, b, allow_nan_stats=False))

    def convertToClasses(self, inputs):
        # inputs is (batch, sequence, 2)
        # class mappings is (assignments, vocab size)
        # outputs is (batch, sequence, 2, assignments)
        batchSize      = tf.shape(inputs)[0]
        sequenceLength = tf.shape(inputs)[1]

        classes = tf.concat([tf.reshape(tf.gather(self.classMappings[i, :], inputs),
                                        (batchSize, sequenceLength, 2, 1))
            for i in range(self.getAssignmentCount())], axis=3)

        return tf.reshape(classes, (batchSize, sequenceLength, 2, self.getAssignmentCount(), 1))

    def expandClassLogitsToVocab(self, classLogits):
        # class logits is (batch size, sequence-length, 2, assignments, class-size)
        # class mappings is (class-assignments, vocab-size)
        # class weights is (class-assignments, vocab-size)
        # output is (batch-size, sequence-length, 2, vocab-size)
        batchSize      = tf.shape(classLogits)[0]
        sequenceLength = tf.shape(classLogits)[1]

        gatheredLogits = tf.concat([tf.reshape(tf.gather(classLogits[:,:,:,i,:], self.classMappings[i, :], axis=3),
                                    (batchSize, sequenceLength, 2, 1, self.vocab.getSize()))
            for i in range(self.getAssignmentCount())], axis=3)

        return tf.reduce_mean(tf.multiply(gatheredLogits, self.classWeights), axis=3)

    def evaluateVocabLoss(self, classLogits, vocabLabels):
        # labels is (batch size, sequence-length, 2)

        batchSize      = tf.shape(classLogits)[0]
        sequenceLength = tf.shape(classLogits)[1]

        sampleCount = self.getSoftmaxSampleCount()
        samples = self.generateSamples(sampleCount)
        sampledLabels = tf.zeros((batchSize, sequenceLength, 2), dtype=tf.int32)

        # sampled mappings is (assignment count, sample count)
        sampledMappings = self.sample(self.classMappings, samples, sampleCount)

        # sampled weights is (assignment count, sample count)
        sampledWeights = self.sample(self.classWeights, samples, sampleCount)

        # gathered logits is (batch size, sequence length, assignment count, sample count)
        gatheredLogits = tf.concat([tf.reshape(tf.gather(classLogits[:,:,:,i,:], sampledMappings[i,:], axis=3),
                                    (batchSize, sequenceLength, 2, 1, sampleCount))
            for i in range(self.getAssignmentCount())], axis=3)

        # gathered weights is (batch size, sequence length, 2, assignment count, sample count)
        gatheredWeights = self.broadcastToExpandedDimension(sampledWeights, batchSize, sequenceLength)

        # gathered logits and weights is (batch size, sequence length, 2, assignment count, sample count + 1)
        gatheredLogits  = self.extendLogits(gatheredLogits, classLogits, vocabLabels)
        gatheredWeights = self.extendWeights(gatheredWeights, vocabLabels)

        # weighted logits is (batch size, sequence length, 2, assignments, sample count + 1)
        weightedLogits = tf.multiply(gatheredLogits, gatheredWeights)

        # vocab logits is (batch size, sequence length, 2, sample count + 1)
        vocabLogits = tf.reduce_mean(weightedLogits, axis=3)

        return self.evaluateLoss(vocabLogits[:, 1:, :, :], sampledLabels[:, 1:, :])

    def generateSamples(self, sampleCount):
        samplesPerAssignment = []

        # TODO: BUG: Dont sample the label
        for assignment in range(self.getAssignmentCount()):
            samples, _, _ = tf.random.uniform_candidate_sampler(
                true_classes=tf.broadcast_to(tf.range(self.vocab.getSize(), dtype=tf.int64),
                                             (1, self.vocab.getSize())),
                num_true=self.vocab.getSize(),
                num_sampled=sampleCount,
                range_max=self.vocab.getSize(),
                unique=True)

            samplesPerAssignment.append(tf.reshape(samples, (1, -1)))

        return tf.concat(samplesPerAssignment, axis=0)

    def extendLogits(self, vocabLogits, classLogits, labels):
        # class logits is (batch size, sequence length, 2, assignment count, sample count)
        # map is (assignment count, vocab size)
        # labels is (batch size, sequence length, 2)
        batchSize      = tf.shape(classLogits)[0]
        sequenceLength = tf.shape(classLogits)[1]

        # labelClasses is (batch size, sequence length, 2, assignment count, 1)
        labelClasses = tf.concat(
            [tf.reshape(tf.gather(self.classMappings[i, :], labels),
                (batchSize, sequenceLength, 2, 1, 1)) for i in range(self.getAssignmentCount())],
            axis=3)

        # gathered logits is (batch size, sequence length, 2, assignment count, 1)
        gatheredLogits = tf.batch_gather(classLogits, labelClasses)

        return tf.concat([gatheredLogits, vocabLogits], axis=4)

    def extendWeights(self, vocabWeights, labels):
        # vocab weights is (batch size, sequence length, 2, assignment count, sample count)
        # labels is (batch size, sequence length)
        batchSize      = tf.shape(vocabWeights)[0]
        sequenceLength = tf.shape(vocabWeights)[1]

        # labelWeights is (batch size, sequence length, 2, assignment count, 1)
        labelWeights = tf.concat(
            [tf.reshape(tf.gather(self.classWeights[i, :], labels),
                (batchSize, sequenceLength, 2, 1, 1)) for i in range(self.getAssignmentCount())],
            axis=3)

        return tf.concat([labelWeights, vocabWeights], axis=4)

    def sample(self, mappings, samples, sampleCount):

        assignments = []

        for i in range(self.getAssignmentCount()):
            assignments.append(tf.reshape(tf.gather(mappings[i, :], samples[i,:]), (1, sampleCount)))

        return tf.concat(assignments, axis=0)

    def broadcastToExpandedDimension(self, tensor, batchSize, sequenceLength):
        classAssignments = tensor.shape[0]
        vocabSize = tensor.shape[1]

        newShape = (batchSize, sequenceLength, 2, classAssignments, vocabSize)

        expandedTensor = tf.broadcast_to(tensor, newShape)

        #print(expandedTensor.shape)

        reshapedTensor = tf.reshape(expandedTensor, newShape)
        #print(reshapedTensor.shape)

        return reshapedTensor

    def runClassModel(self, inputs):
        #print("inputs", inputs.shape)

        inputEmbeddings = self.convertToEmbeddings(inputs)

        #print("inputEmbeddings", inputEmbeddings.shape)

        # run encoder (logits is (batch-size, sequence-length, assignments, class-count))
        encodedEmbeddings = self.runEncoder(inputEmbeddings)

        logits = self.runDecoder(encodedEmbeddings)

        #print("logits", logits.shape)

        return logits

    def runClassificationModel(self):
        batchSize = tf.shape(self.features)[0]
        sequenceLength = tf.shape(self.features)[1]

        features = tf.reshape(self.features, (batchSize, sequenceLength, 2,
            self.getAssignmentCount(), self.getEmbeddingSize()))

        features = self.multiheadedAttention(features)

        # features is (batch-size, sequence-length, 2, assignments, embedding-size)
        reducedFeatures = tf.reduce_max(features, axis=1)

        # reducedFeatures is (batch size, 2, assignments, embedding-size)
        transposedFeatures = tf.transpose(reducedFeatures, [0,2,1,3])

        # transposedFeatures is (batch size, assignments, 2, embedding-size)
        reshapedFeatures = tf.reshape(transposedFeatures, (-1, self.getAssignmentCount(),
            2 * self.getEmbeddingSize()))

        return tf.layers.dense(reshapedFeatures, units=2)

    def convertToEmbeddings(self, sequenceIds):
        assignments = []
        for assignment in range(self.getAssignmentCount()):
            assignments.append(self.convertToClassEmbeddings(sequenceIds, assignment))

        return tf.concat(assignments, axis = 3)

    def convertToClassEmbeddings(self, ids, assignment):

        with tf.variable_scope("linear-embeddings", reuse=tf.AUTO_REUSE):
            wordEmbeddingsGlobal = tf.get_variable('class-embeddings-' + str(assignment), \
                    [self.getNumberOfClasses(), self.getEmbeddingSize()])

        wordEmbeddings = tf.nn.embedding_lookup(wordEmbeddingsGlobal, ids[:, :, :, assignment, :])
        return wordEmbeddings

    def runEncoder(self, embeddings):
        return self.multiheadedAttentionStack(embeddings)

    def runDecoder(self, embeddings):

        batchSize      = tf.shape(embeddings)[0]
        sequenceLength = tf.shape(embeddings)[1]
        # embeddings is (batch size, sequence length, 2, assignments, classes)
        return tf.concat([tf.reshape(tf.layers.dense(embeddings[:,:,:,i,:], units=self.getNumberOfClasses()),
                (batchSize, sequenceLength, 2, 1, self.getNumberOfClasses()))
            for i in range(self.getAssignmentCount())], axis=3)

    def multiheadedAttentionStack(self, embeddings):

        embeddings = self.addPositions(embeddings)

        # embeddings (batch-size, sequence-length, 2, assignments, hidden-dimension)
        for layer in range(self.getNumberOfLayers()):
            embeddings = self.multiheadedAttention(embeddings)

            if self.isMiddleLayer(layer):
                batchSize      = tf.shape(embeddings)[0]
                sequenceLength = tf.shape(embeddings)[1]

                self.features = tf.identity(tf.reshape(embeddings, (batchSize, sequenceLength, 2,
                    self.getAssignmentCount() * self.getEmbeddingSize())), name="features")

        return embeddings

    def addPositions(self, embeddings):
        batchSize      = tf.shape(embeddings)[0]
        sequenceLength = tf.shape(embeddings)[1]

        halfSequenceLength = (sequenceLength + 1) // 2

        positions = tf.cast(tf.reshape(tf.range(halfSequenceLength),
            (1, halfSequenceLength, 1, 1, 1)), dtype=tf.float32)
        dimensions = tf.cast(tf.reshape(tf.range(self.getEmbeddingSize()),
            (1, 1, 1, 1, self.getEmbeddingSize())), dtype=tf.float32)

        angles = positions / tf.pow(2.0 * tf.cast(halfSequenceLength, dtype=tf.float32),
                                    2.0 * dimensions / self.getEmbeddingSize())

        evenPositionEmbeddings = tf.reshape(tf.sin(angles),
            (1, halfSequenceLength, 1, 1, 1, self.getEmbeddingSize()))
        oddPositionEmbeddings  = tf.reshape(tf.cos(angles),
            (1, halfSequenceLength, 1, 1, 1, self.getEmbeddingSize()))

        # merge them
        positionEmbeddings = tf.concat([evenPositionEmbeddings, oddPositionEmbeddings], axis=2)
        positionEmbeddings = tf.reshape(positionEmbeddings,
            (1, 2 * halfSequenceLength, 1, 1, self.getEmbeddingSize()))

        positionEmbeddings = positionEmbeddings[:, 0:sequenceLength, :, :, :]

        return embeddings + positionEmbeddings

    def isMiddleLayer(self, layer):
        if self.getNumberOfLayers() > 1:
            return layer == (self.getNumberOfLayers() - 2)

        return layer == (self.getNumberOfLayers() - 1)

    def multiheadedAttention(self, embeddings):
        # embeddings (batch-size, sequence-length, assignments, hidden-dimension)
        projectedEmbeddings = self.projectEmbeddings(embeddings)

        # proj-embeddings (batch-size, sequence-length, assignments, QKV, attention-heads, hidden-dimension)
        attentionOutput = self.runAttention(projectedEmbeddings)

        # project back
        outputEmbeddings = self.projectBackEmbeddings(attentionOutput)

        # add and norm
        embeddings = self.addAndNorm(outputEmbeddings, embeddings)

        # dense layer
        denseOutput = tf.layers.dense(embeddings,
            self.getEmbeddingSize(), activation="relu")

        # add and norm
        denseOutput = self.addAndNorm(denseOutput, embeddings)

        return denseOutput

    def projectEmbeddings(self, embeddings):
        output = tf.layers.dense(embeddings,
            embeddings.shape[-1] * 3 * self.getNumberOfAttentionHeads())

        batchSize      = tf.shape(embeddings)[0]
        sequenceLength = tf.shape(embeddings)[1]
        assignments    = embeddings.shape[3]

        return tf.reshape(output,
            (batchSize, sequenceLength, 2, assignments, 3,
             self.getNumberOfAttentionHeads(), embeddings.shape[-1]))

    def projectBackEmbeddings(self, embeddings):
        # embeddings are (batch-size, sequence-length, 2, assignments, attention-heads, embedding-size)
        # project to (batch-size, sequece-length, 2, assignments, embedding-size)

        batchSize      = tf.shape(embeddings)[0]
        sequenceLength = tf.shape(embeddings)[1]
        assignments    = embeddings.shape[3]

        reshapedEmbeddings = tf.reshape(embeddings, (batchSize, sequenceLength, 2, assignments,
            embeddings.shape[-1] * embeddings.shape[-2]))

        projectedEmbeddings = tf.layers.dense(reshapedEmbeddings, self.getEmbeddingSize())

        return projectedEmbeddings

    def addAndNorm(self, left, right):
        return tf.contrib.layers.layer_norm(tf.add(left, right))

    def runAttention(self, embeddings):
        # Q,K,V (batch-size, sequence-length, 2, assignments, attention-heads, hidden-dimension)
        Q = embeddings[:,:,:,:,0,:,:]
        K = embeddings[:,:,:,:,1,:,:]
        V = embeddings[:,:,:,:,2,:,:]

        readOn = tf.matmul(Q, K, transpose_b=True)

        scale = math.sqrt(self.getEmbeddingSize())

        scaledReadOn = readOn / scale

        contribution = tf.nn.softmax(scaledReadOn, axis=1)

        result = tf.matmul(contribution, V)

        return result

    def checkpoint(self, prefix):
        """Creates a checkpoint of the current model and saves to model
        directory.
        """

        self.checkpointer.setPrefix(prefix)
        directory = self.checkpointer.getModelSaveDirectory()
        logger.debug("Saving checkpoint to: " + str(directory))

        self.checkpointer.checkpoint()

        with self.graph.as_default():
            tf.saved_model.simple_save(self.session,
                directory,
                inputs={"input-tokens" : self.inputTokens},
                outputs={"output-probabilities" : self.outputProbabilities})

        self.checkpointer.cleanup()


    """Functions to load configuration parameters."""
    def getEmbeddingSize(self):
        return int(self.config["model"]["embedding-size"])

    def getAssignmentCount(self):
        return int(self.config["model"]["assignment-count"])

    def getSoftmaxSampleCount(self):
        return int(self.config["model"]["softmax-sample-count"])

    def getNumberOfClasses(self):
        return int(self.config["model"]["number-of-classes"])

    def getNumberOfDirectClasses(self):
        return int(self.config["model"]["number-of-direct-classes"])

    def getNumberOfLayers(self):
        return int(self.config["model"]["number-of-layers"])

    def getNumberOfAttentionHeads(self):
        return int(self.config["model"]["number-of-attention-heads"])

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

    def getStepsPerTensorboardLog(self):
        return int(self.config["model"]["steps-per-tensorboard-log"])

    def getValidationStepsPerEpoch(self):
        return int(self.config["model"]["validation-steps-per-epoch"])

    def getExperimentDirectory(self):
        return self.config["model"]["directory"]











