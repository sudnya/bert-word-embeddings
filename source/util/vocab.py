
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

def addToken(vocab, tokenBuffer):
    if len(tokenBuffer) == 0:
        return

    token = ''.join(tokenBuffer)

    if not token in vocab:
        vocab[token] = len(vocab)

    del tokenBuffer[:]

def isSingleCharacterToken(character):
    return not character.isalnum()

def saveVocab(dataset, size, directory):
    import os
    vocab = createInitialVocab()

    outputPath = os.path.join(directory, "vocab.txt")

    tokenBuffer = []

    for i in range(size):
        nextCharacter = dataset.next()

        if isSingleCharacterToken(nextCharacter):
            addToken(vocab, tokenBuffer)
            tokenBuffer.append(nextCharacter)
            addToken(vocab, tokenBuffer)
            continue

        tokenBuffer.append(nextCharacter)
        possibleToken = ''.join(tokenBuffer)
        if isToken(possibleToken):
            addToken(vocab, tokenBuffer)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(outputPath, "w", encoding='utf-8') as outputFile:
        for token in vocab.keys():
            if token[-1] != '\n':
                token += '\n'
            outputFile.write(token)


