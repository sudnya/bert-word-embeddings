'''
Embedding layer
Encoder
Decoder

'''
class BERTModel:
    def __init__(self, config):
        self.config = config

    def train(self):
        pass

    def runEncoderDecoder(self, inputSequence, historicSequence):
        # inputseq -> tensor : sx1
        # inputSeq -> hidden 1 x embedding x hidden : k1xs
        # k1xs -> encoder -> k2xs
        # historicSeq -> tensor: sx1
        # k2xs, historicSeq -> (decoder) -> nextWord (1xv)

