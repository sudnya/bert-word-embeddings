
import logging

logger = logging.getLogger(__name__)

def isWhitespace(character):
    return character.isspace()

def isToken(token):
    if len(token) == 1:
        return False

    if isWhitespace(token[-1]):
        return True

    return False

def isSingleCharacterToken(character):
    return not character.isalnum()

class UnlimitedVocabTokenizerAdaptor:
    def __init__(self, source):
        self.source = source
        self.tokens = []
        self.maximumSize = None
        self.tokenCount = None

    def next(self):

        if self.hasTokens():
            return self.pop()

        tokenBuffer = []

        while True:
            nextCharacter = self.source.next()

            if isSingleCharacterToken(nextCharacter):
                self.addToken(tokenBuffer)
                tokenBuffer.append(nextCharacter)
                self.addToken(tokenBuffer)
                break

            tokenBuffer.append(nextCharacter)
            possibleToken = ''.join(tokenBuffer)
            if isToken(possibleToken):
                addToken(vocab, tokenBuffer)
                break

        return self.pop()

    def pop(self):
        nextToken = self.tokens[0]

        del self.tokens[0]

        return nextToken

    def hasTokens(self):
        return len(self.tokens) > 0

    def addToken(self, tokenBuffer):
        if len(tokenBuffer) == 0:
            return
        self.tokens.append(''.join(tokenBuffer))

        del tokenBuffer[:]

    def size(self):
        if self.tokenCount is None:
            self.tokenCount = self.getTokenCount()
            self.reset()
        return self.tokenCount

    def reset(self):
        self.source.reset()

    def setMaximumSize(self, size):
        self.maximumSize = size

    def getTokenCount(self):
        count = 0

        logger.debug("Scanning token count...")

        try:
            while True:
                token = self.next()
                if len(token) == 0:
                    break
                count += 1
                if count % 1e6 == 0:
                    logger.info(" " + str(count))

                if not self.maximumSize is None:
                    if count >= self.maximumSize:
                        break
        except ValueError:
            pass

        logger.debug("Scanning token count..." + str(count))

        return count






