import tensorflow as tf

from ..base_runner import BaseRunner


class MomentumAlgorithmRunner(BaseRunner):
    def _get_optimizer(self):
        return tf.keras.optimizers.SGD(**self.hyper_parameters)
