
from inference.Featurizer import Featurizer
from models.Vocab import Vocab
from sklearn.cluster import MiniBatchKMeans as MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA as IncrementalPCA

import os
import numpy
import statistics

import logging

logger = logging.getLogger(__name__)

class Clusterer:
    def __init__(self, config, validationDataset, outputDirectory, numberOfClusters):
        self.config = config
        self.validationDataset = validationDataset
        self.numberOfClusters = numberOfClusters
        self.outputDirectory = outputDirectory

    def groupDataIntoClusters(self):

        kmeans = MiniBatchKMeans(n_clusters=self.numberOfClusters)
        featurizer = Featurizer(self.config, self.validationDataset)
        vocab = Vocab(self.config)

        if self.usePCA():
            pca = IncrementalPCA(n_components=32)

        logger.info("Reducing dimensionality...")

        # fit the pca model
        if self.usePCA():
            for iteration in range(self.getIterations()):
                if iteration % 10 == 0:
                    logger.info(" " + str(iteration) + " / " + str(self.getIterations()))
                inputs, labels, embeddings = featurizer.featurizeOneBatch()

                pca.partial_fit(numpy.reshape(embeddings, (-1, embeddings.shape[-1])))

            self.validationDataset.reset()

        logger.info("Fitting model...")

        # fit the kmeans model
        for iteration in range(self.getIterations()):
            if iteration % 10 == 0:
                logger.info(" " + str(iteration) + " / " + str(self.getIterations()))
            inputs, labels, embeddings = featurizer.featurizeOneBatch()

            if self.usePCA():
                embeddings = pca.transform(numpy.reshape(embeddings, (-1, embeddings.shape[-1])))

            kmeans.partial_fit(numpy.reshape(embeddings, (-1, embeddings.shape[-1])))

        self.validationDataset.reset()

        # group into clusters
        # create a histogram of word frequencies per cluster
        clusterHistogram = { i : {} for i in range(self.numberOfClusters) }
        clusterWins = { i : 0 for i in range(self.numberOfClusters) }
        documentMap = {}

        logger.info("Clustering data...")

        for iteration in range(self.getIterations()):
            if iteration % 10 == 0:
                logger.info(" " + str(iteration) + " / " + str(self.getIterations()))
            inputs, labels, embeddings = featurizer.featurizeOneBatch()

            chunkLength = embeddings.shape[1]
            batchSize = embeddings.shape[0]

            if self.usePCA():
                embeddings = pca.transform(numpy.reshape(embeddings, (-1, embeddings.shape[-1])))

            clusters = numpy.reshape(
                kmeans.predict(numpy.reshape(embeddings, (-1, embeddings.shape[-1]))),
                (batchSize, chunkLength))

            for batch in range(batchSize):
                documentId = labels[batch, 0]

                if not documentId in documentMap:
                    documentMap[documentId] = []

                clusterIds = []

                for wordIndex in range(1, chunkLength):

                    word = vocab.getTokenString(labels[batch, wordIndex])
                    cluster = clusters[batch, wordIndex]

                    clusterIds.append(cluster)

                    if not labels[batch, wordIndex] in clusterHistogram[cluster]:
                        clusterHistogram[cluster][labels[batch, wordIndex]] = 0

                    clusterHistogram[cluster][labels[batch, wordIndex]] += 1
                    clusterWins[cluster] += 1

                documentMap[documentId].extend(clusterIds)

        if not os.path.exists(self.outputDirectory):
            os.makedirs(self.outputDirectory)

        # write histograms
        with open(self.getOutputHistogramFileName(), "w") as log:
            for clusterId, count in sorted(clusterWins, key=lambda x : x[1], reverse=True):
                words = clusterHistograms[clusterId]
                log.write("Cluster, " + str(clusterId) + "\n")
                for wordIndex, count in sorted(words.items(), key=lambda x : x[1], reverse=True):
                    log.write("    '" + vocab.getTokenString(wordIndex) +
                          "' " + str(count) + "\n")

        # write document clusters
        for documentId, clusters in documentMap.items():

            histogram = {}

            for cluster in clusters:
                if not cluster in histogram:
                    histogram[cluster] = 0

                histogram[cluster] += 1

            with open(self.getOutputDocumentClusterFileName(documentId), "w") as log:

                for cluster, count in sorted(histogram.items(), key=lambda x:x[1], reverse=True):

                    words = clusterHistogram[cluster]
                    topWord = vocab.getTokenString(sorted(words.items(), key=lambda x : x[1], reverse=True)[0][0])
                    log.write("Cluster, " + str(cluster) + ", " + topWord + ", " + str(count) + "\n")

    def vote(self, array):
        return statistics.mode(array)

    def getOutputFileName(self):
        return os.path.join(self.outputDirectory, "clusters.txt")

    def getOutputClusterFileName(self, cluster):
        return os.path.join(self.outputDirectory, "id-" + str(cluster) + "-cluster.txt")

    def getOutputDocumentClusterFileName(self, documentId):
        return os.path.join(self.outputDirectory, "document-id-" + str(documentId) + "-histogram.txt")

    def getOutputHistogramFileName(self):
        return os.path.join(self.outputDirectory, "histogram.txt")

    def getIterations(self):
        if "iterations" in self.config["predictor"]:
            count = int(self.config["predictor"]["iterations"])
        elif "validationStepsPerEpoch" in self.config["model"]:
            count = int(self.config["model"]["validationStepsPerEpoch"])
        else:
            count = int(self.config["model"]["validation-steps-per-epoch"])

        self.validationDataset.setMaximumSize(count)

        return self.validationDataset.size()

    def usePCA(self):
        if "use-pca" in self.config["predictor"]:
            return bool(self.config["predictor"]["use-pca"])

        return False












