import keras
import tensorflow_addons as tfa

from ..base_runner import BaseRunner


class LookaheadAlgorithmRunner(BaseRunner):
    def _get_optimizer(self):
        return tfa.optimizers.Lookahead(keras.optimizers.SGD(), **self.hyper_parameters)
