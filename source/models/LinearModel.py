# -*- coding: utf-8 -*-
import logging
import numpy
import os
import json
import shutil
import time
import tensorflow as tf

from models.Vocab import Vocab
from models.ModelDescriptionCheckpointer import ModelDescriptionCheckpointer


logger = logging.getLogger(__name__)

"""Implements a linear dense model using Tensorflow"""
class LinearModel:
    def __init__(self, config, trainingDataSource, validationDataSource):
        """Initializes the linear model object.

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
        self.checkpointer = ModelDescriptionCheckpointer(config, "LinearModel")
        


    def train(self):
        """Trains the linear model.

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
        self.checkpointer.load()

        directory = self.checkpointer.getModelDirectory()

        logger.debug("Loading checkpoint from: " + str(directory))
        assert False, "not implemented"



    def createModel(self):
        ## (batch, sequence-length, 1)
        self.inputTokens = tf.placeholder(tf.int32, shape=(None, None), 
                name="input-tokens")

        self.labels = tf.placeholder(tf.int32, shape=(None, None), 
                name="output-labels")

        batchOutputs = self.processInputMiniBatch(self.inputTokens)
        self.predictedLabels = batchOutputs
        self.loss = self.evaluateLoss(batchOutputs, self.labels)

        # optimizer
        self.optimizerStep = self.createOptimizerStep(self.loss)

        # initializers
        self.globalInitializer = tf.global_variables_initializer()
        self.localInitializer  = tf.local_variables_initializer()

        # summaries
        self.setupSummaries()

        # do the initialization
        self.initializeModel()


    def initializeModel(self):
        self.session.run(self.globalInitializer)
        self.session.run(self.localInitializer)


    def runOnTrainingDataset(self, epoch):
        """Trains the linear model on the training dataset for one epoch."""
        trainStart = time.time()

        for step in range(self.getStepsPerEpoch()):
            generatorStart = time.time()

            inputs, labels = self.trainingDataSource.next()

            generatorEnd = time.time()

            trainStepStart = time.time()
            loss, gradNorm = self.trainingStep(inputs, labels, step)
            trainStepEnd = time.time()

            message = ("Epoch (" + str(epoch) + " / " + str(self.getEpochs()) +
                "), Step (" + str(step) + " / " + str(self.getStepsPerEpoch()) +
                "), Generator time: " + ("%.2f" % (generatorEnd - generatorStart)) +
                ", training step time: " + ("%.2f" % (trainStepEnd -
                    trainStepStart) +
                ", loss: " + str("%.2f" % loss) +
                ", grad norm: " + str("%.2f" % gradNorm)))

            print(message, end="\r", flush=True)

        trainEnd = time.time()

        print(message)
        logger.debug(" Training took: " + (str(trainEnd - trainStart)) + " seconds...")



    def trainingStep(self, inputs, labels, step):
        """Training step for one minibatch of training data."""
        inputs = numpy.array(inputs)
        labels = numpy.array(labels)

        trainingLoss, gradNorm, summaries, _ = self.session.run([self.loss,
            self.gradientNorm, self.mergedSummary, self.optimizerStep], 
            feed_dict={self.inputTokens : inputs, self.labels : labels })
        
        self.trainingSummaryWriter.add_summary(summaries, step)
        return trainingLoss, gradNorm



    def runOnValidationDataset(self, epoch):
        """Runs the linear model on the validation dataset for one epoch."""
        
        validationStart = time.time()

        for step in range(self.getValidationStepsPerEpoch()):
            generatorStart = time.time()

            inputs, labels = self.validationDataSource.next()

            generatorEnd = time.time()

            validationStepStart = time.time()
            loss, summary = self.validationStep(inputs, labels)
            validationStepEnd = time.time()

            message = ("Validation Step (" + str(step) + " / " +
                    str(self.getValidationStepsPerEpoch()) +
                "), Generator time: " + ("%.2f" % (generatorEnd - generatorStart)) +
                ", validation step time: " + ("%.2f" % (validationStepEnd -
                    validationStepStart) +
                ", loss: " + str(loss)))

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
            learning_rate=float(self.config["model"]["learningRate"]),
            beta1=0.9,
            beta2=0.999,
            epsilon=numpy.finfo(float).eps,
            name="optimizer-step")

        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients,
        self.config["model"]["gradientClippingFactor"])
        self.gradientNorm = tf.global_norm(gradients, name="gradient-norm")

        return optimizer.apply_gradients(zip(gradients, variables))


    def setupSummaries(self):
        tf.summary.scalar('cross-entropy', self.loss)
        tf.summary.scalar('gradient-norm', self.gradientNorm)

        self.mergedSummary = tf.summary.merge_all()

        self.trainingSummaryWriter = tf.summary.FileWriter(
            os.path.join(self.getExperimentDirectory(), 'training-summaries'),
            self.graph)

        if self.shouldRunValidation():
            self.validationSummaryWriter = tf.summary.FileWriter(
                os.path.join(self.getExperimentDirectory(), 'validation-summaries'),
                self.graph)


    def evaluateLoss(self, batchOutputs, labels):
        return tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=batchOutputs
        )


    def processInputMiniBatch(self, inputs):
        return self.runEncoderDecoder(inputs, inputs)


    def runEncoderDecoder(self, inputSequence, historicSequence):
        # convert sequences to embeddings (output embeddings are Tensor(batch-size, sequence-length, hidden))
        inputEmbeddings   = self.convertToEmbeddings(inputSequence)
        historicEmbeddings = self.convertToEmbeddings(historicSequence)

        # run encoder (encodedEmbeddings is (batch-size, sequence-length, hidden))
        encodedEmbeddings = self.runEncoder(inputEmbeddings)

        # run decoder (logits is Tensor(batch-size, sequence-length, vocab-size)
        logits = self.runDecoder(encodedEmbeddings, historicEmbeddings)
        
        return logits


    def convertToEmbeddings(self, sequenceIds):
        with tf.variable_scope("linear-embeddings", reuse=tf.AUTO_REUSE):
            wordEmbeddingsGlobal = tf.get_variable('word-embeddings', \
                    [self.vocab.getSize(), self.getEmbeddingSize()])
        wordEmbeddings = tf.nn.embedding_lookup(wordEmbeddingsGlobal, sequenceIds)
        return wordEmbeddings


    def runEncoder(self, embeddings):
        return tf.layers.dense(embeddings, self.getEmbeddingSize(),
        activation="relu")


    def runDecoder(self, inputEmbeddings, historicEmbeddings):
        return tf.layers.dense(tf.concat([inputEmbeddings, historicEmbeddings],
            axis=2), self.vocab.getSize())


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
                outputs={"predictions" : self.predictedLabels})

        if exists:
            shutil.rmtree(tempDirectory)
    
    
    """Functions to load configuration parameters."""
    def getEmbeddingSize(self):
        return int(self.config["model"]["embeddingSize"])

    
    def shouldRunValidation(self):
        return self.config["model"]["runValidation"]

    
    def getEpochs(self):
        return int(self.config["model"]["epochs"])
    
    
    def getStepsPerEpoch(self):
        return int(self.config["model"]["stepsPerEpoch"])
    
    
    def getValidationStepsPerEpoch(self):
        return int(self.config["model"]["validationStepsPerEpoch"])
    
    
    def getExperimentDirectory(self):
        return self.config["model"]["directory"]
