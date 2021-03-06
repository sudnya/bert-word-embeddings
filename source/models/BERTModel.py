# -*- coding: utf-8 -*-
import logging
import math
import numpy
import os
import json
import shutil
import time
import tensorflow as tf

from models.Vocab import Vocab
from models.ModelDescriptionCheckpointer import ModelDescriptionCheckpointer


logger = logging.getLogger(__name__)

class BERTModel:
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
        self.checkpointer = ModelDescriptionCheckpointer(config, "BERTModel")
        self.isLoaded = False
        


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
        ## (batch, sequence-length, 1)
        self.inputTokens = tf.placeholder(tf.int32, shape=(None, None), 
                name="input-tokens")

        self.labels = tf.placeholder(tf.int32, shape=(None, None), 
                name="output-labels")

        predictedLogits = self.processInputMiniBatch(self.inputTokens)
        self.loss = self.evaluateLoss(predictedLogits, self.labels)
        self.outputProbabilities = tf.nn.softmax(predictedLogits,
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
        return tf.identity(tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=batchOutputs
        ), name="loss")


    def processInputMiniBatch(self, inputs):
        return self.runBERT(inputs, inputs)


    def runBERT(self, inputSequence, historicSequence):
        # convert sequences to embeddings (output embeddings are Tensor(batch-size, sequence-length, hidden))
        inputEmbeddings   = self.convertToEmbeddings(inputSequence)
        inputEmbeddingsPositionallyEncoded = self.getPositionalEncodings(inputEmbeddings)

        # run encoder (encodedEmbeddings is (batch-size, sequence-length, hidden))
        encodedEmbeddings = self.runEncoder(inputEmbeddingsPositionallyEncoded)
        return tf.layers.dense(encodedEmbeddings, units=self.vocab.getSize())


    def convertToEmbeddings(self, sequenceIds):
        with tf.variable_scope("linear-embeddings", reuse=tf.AUTO_REUSE):
            wordEmbeddingsGlobal = tf.get_variable('word-embeddings', \
                    [self.vocab.getSize(), self.getEmbeddingSize()])
        wordEmbeddings = tf.nn.embedding_lookup(wordEmbeddingsGlobal, sequenceIds)
        return wordEmbeddings


    def getPositionalEncodings(self, inputEmbeddings):
        #PE(pos,2i)=sin(pos/100002i/dmodel)PE(pos,2i)=sin(pos/100002i/dmodel) 
        #PE(pos,2i+1)=cos(pos/100002i/dmodel)PE(pos,2i+1)=cos(pos/100002i/dmodel) where pospos is the position and ii is the dimension.
        batchSize       = tf.shape(inputEmbeddings)[0]
        sequenceLength  = tf.shape(inputEmbeddings)[1]
        hiddenDimension = tf.shape(inputEmbeddings)[2]
        
        sequenceRange = tf.reshape(tf.range(tf.cast(sequenceLength, tf.float32)), (1, sequenceLength, 1))
        hiddenRange = tf.reshape(tf.range(tf.cast(hiddenDimension, tf.float32)), (1, 1, hiddenDimension))
        rawPE = sequenceRange / (tf.pow(10000.0, 2.0 * hiddenRange /
                tf.cast(hiddenDimension, tf.float32)))

        PE_cos = tf.cos(rawPE[:,0::2,:])
        PE_sin = tf.sin(rawPE[:,1::2,:])
        
        PE_cos = tf.reshape(PE_cos, (batchSize, tf.shape(PE_cos)[1], 1, hiddenDimension))
        PE_sin = tf.reshape(PE_sin, (batchSize, tf.shape(PE_sin)[1], 1, hiddenDimension))
        
        PE = tf.concat([PE_cos, PE_sin], axis=2)

        return inputEmbeddings + tf.reshape(PE, (batchSize, sequenceLength, hiddenDimension))

    def runEncoder(self, embeddings):
        for i in range(self.getLayerCount()):
            right = self.multiHeadedAttention(embeddings)
            left = tf.layers.dense(right, units=embeddings.shape[-1])
            embeddings = self.addAndNorm(left, right)
        return embeddings


    def multiHeadedAttention(self, embeddings):
        # Q,K,V are all -> projected embeddings
        projectedEmbeddings = self.projectEmbeddings(embeddings)
        attentionResults = self.attention(projectedEmbeddings)
        left = self.projectAttentionOutput(attentionResults)
        return self.addAndNorm(left, embeddings)

    def projectEmbeddings(self, embeddings):
        #input -> m, seqL, embedding size
        #output -> m, seqL, 3 * numberOfAttentionHeads * embedding size
        retVal = tf.layers.dense(embeddings,
                units=3*self.getAttentionHeads()*embeddings.shape[-1])

        return tf.reshape(retVal, [tf.shape(retVal)[0], tf.shape(retVal)[1], 3, 
                self.getAttentionHeads(), embeddings.shape[-1]])


    def attention(self, projectedEmbeddings):
        #m, seqL, (Q, K, V), attentionHeads, embedding size
        Q = projectedEmbeddings[:,:,0,:,:]
        K = projectedEmbeddings[:,:,1,:,:]
        V = projectedEmbeddings[:,:,2,:,:]
        d_k = int(projectedEmbeddings.shape[-1])
        
        m1 = tf.matmul(Q, K, transpose_b=True) / math.sqrt(d_k)
        smx = tf.nn.softmax(m1)
        return tf.matmul(smx, V)

    def projectAttentionOutput(self, attentionResults):
        #attentionResults -> m, seqL, attentionHeads, embedding size
#new shape is (batch, sequence length, heads * embedding-size)
        batchSize = tf.shape(attentionResults)[0]
        sequenceLength = tf.shape(attentionResults)[1]

        reshapedEmbeddings = tf.reshape(attentionResults, 
                (batchSize, sequenceLength,
                        attentionResults.shape[-1]*attentionResults.shape[-2]))
        return tf.layers.dense(reshapedEmbeddings,
                units=attentionResults.shape[-1])


    def addAndNorm(self, left, right):
        normalizedLeft = tf.contrib.layers.layer_norm(left)
        return tf.add(normalizedLeft, right)


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
        return int(self.config["model"]["embeddingSize"])

    
    def shouldRunValidation(self):
        return self.config["model"]["runValidation"]

    
    def getEpochs(self):
        return int(self.config["model"]["epochs"])
    

    def getShouldCreateModel(self):
        if not "createNewModel" in self.config["model"]:
            return False
        return bool(self.config["model"]["createNewModel"])

    
    def getStepsPerEpoch(self):
        return int(self.config["model"]["stepsPerEpoch"])
    
    
    def getValidationStepsPerEpoch(self):
        return int(self.config["model"]["validationStepsPerEpoch"])
    
    def getLayerCount(self):
        return self.config["model"]["layerCount"]

    def getAttentionHeads(self):
        return self.config["model"]["attentionHeads"]
    
    def getExperimentDirectory(self):
        return self.config["model"]["directory"]


