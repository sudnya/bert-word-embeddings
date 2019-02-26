
from models.Vocab import Vocab

def createInitialVocab():
    vocab = {}

    # handle all of ascii
    for i in range(127):
        vocab[chr(i)] = len(vocab) + Vocab.getVocabOffset()

    return vocab

def saveVocab(dataset, size, directory):
    import os

    vocab = createInitialVocab()

    outputPath = os.path.join(directory, "vocab.txt")

    for i in range(size):
        string = dataset.next()
        assert len(string) > 0
        if not string in vocab:
            vocab[string] = len(vocab) + Vocab.getVocabOffset()

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(outputPath, "w", encoding='utf-8') as outputFile:
        for token in vocab.keys():
            if token[-1] != '\n':
                token += '\n'
            outputFile.write(token)


