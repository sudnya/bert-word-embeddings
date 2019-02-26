
from inference.FallbackTokenEvaluator import FallbackTokenEvaluator
from inference.PerTokenEvaluator import PerTokenEvaluator

import logging

logger = logging.getLogger(__name__)

class EvaluatorFactory:
    def __init__(self, config):
        self.config = config

    def create(self):
        logger.debug("Creating evaluator for dataset.")

        if self.hasFallbackTokenizer():
            logger.debug(" fallback token evaluator")
            return FallbackTokenEvaluator(self.config)

        if self.hasUnkTokenizer():
            logger.debug(" per token evaluator")
            return PerTokenEvaluator(self.config)

        assert False, "Cant figure out evaluator type...."

    def hasFallbackTokenizer(self):
        if not "adaptor" in self.config:
            return False

        if not "tokenizer" in self.config["adaptor"]:
            return False

        return True

    def hasUnkTokenizer(self):
        if not "adaptor" in self.config:
            return False

        if not "unk-tokenizer" in self.config["adaptor"]:
            return False

        return True


