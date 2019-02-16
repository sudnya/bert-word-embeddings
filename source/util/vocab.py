
def createInitialVocab():
    vocab = {}

    # handle all of ascii
    for i in range(127):
        vocab[chr(i)] = len(vocab)

    return vocab

def isWhitespace(character):
    return character.isspace()

def isToken(token):
    if len(token) == 1:
        return False

    if isWhitespace(token[-1]):
        return True

    return False

def saveVocab(dataset, size, directory):
    import os
    vocab = createInitialVocab()

    outputPath = os.path.join(directory, "vocab.txt")

    tokenBuffer = []

    for i in range(size):
        tokenBuffer.append(dataset.next())
        token = ''.join(tokenBuffer)
        if isToken(token):
            tokenBuffer = []
            if not token in vocab:
                vocab[token] = len(vocab)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(outputPath, "w", encoding='utf-8') as outputFile:
        for token in vocab.keys():
            if token[-1] != '\n':
                token += '\n'
            outputFile.write(token)


