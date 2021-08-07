import tensorflow as tf
import tensorflow_addons as tfa

from ..base_runner import BaseRunner


class LookaheadAlgorithmRunner(BaseRunner):
    def _get_optimizer(self):
        return tfa.optimizers.Lookahead(tf.keras.optimizers.SGD(), **self.hyper_parameters)
