
from models.Vocab import Vocab

def createInitialVocab():
    vocab = {}

    # handle all of ascii
    for i in range(127):
        vocab[chr(i)] = 0

    return vocab

import logging
logger = logging.getLogger(__name__)

def saveVocab(dataset, size, directory):
    import os
    import time

    vocab = createInitialVocab()

    if os.path.isdir(directory):
        outputPath = os.path.join(directory, "vocab.txt")

        if not os.path.exists(directory):
            os.makedirs(directory)
    else:
        outputPath = directory

    previousVocabSize = 0

    start = time.time()
    totalTokens = 0

    while True:
        string = dataset.next()
        if len(string) == 0:
            break
        if not string in vocab:
            vocab[string] = 0

        totalTokens += 1
        vocab[string] += 1

        if len(vocab) + Vocab.getVocabOffset() >= previousVocabSize + size * 0.01:
            previousVocabSize = len(vocab) + Vocab.getVocabOffset()
            logger.debug("Vocab size is " + str(previousVocabSize) + " time so far: " +
                str(time.time() - start) + " total tokens: "  + str(totalTokens))

        if len(vocab) + Vocab.getVocabOffset() >= size:
            break

    with open(outputPath, "w", encoding='utf-8') as outputFile:
        for token, count in reversed(sorted(vocab.items(), key=lambda x: x[1])):
            if token[-1] != '\n':
                token += '\n'
            outputFile.write(token)


