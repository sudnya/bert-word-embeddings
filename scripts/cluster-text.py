

import sys
sys.path.append('source')

from models.Vocab import Vocab
from sklearn.cluster import MiniBatchKMeans as MiniBatchKMeans

import os
import numpy

directory = 'output-features-16k-classes-2-layers-200MB-3'
numberOfClusters = 16

vocab = Vocab({ "model" : { "vocab":os.path.join(directory, 'vocab.txt') } })

embeddings = numpy.load(os.path.join(directory, 'features.npy'))
inputs = numpy.load(os.path.join(directory, 'inputs.npy'))
labels = numpy.load(os.path.join(directory, 'labels.npy'))

chunkCount = embeddings.shape[0]
chunkLength = embeddings.shape[1]

clusters = numpy.reshape(KMeans(n_clusters=numberOfClusters).fit_predict(numpy.reshape(embeddings, (-1, embeddings.shape[-1]))), (chunkCount, chunkLength))

clusterMap = {i : [] for i in range(numberOfClusters)}

for chunk in range(chunkCount):
    chunkString = [vocab.getTokenString(labels[chunk, word]) for word in range(chunkLength)]

    for word in range(chunkLength):
        clusterId = clusters[chunk, word]
        wordString = vocab.getTokenString(labels[chunk, word])

        clusterMap[clusterId].append((wordString, chunkString))


for clusterId, words in clusterMap.items():
    print("Cluster", clusterId)
    for word, chunk in words:
        print(" ", "'" + word + "'", chunk)




