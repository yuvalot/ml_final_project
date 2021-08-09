import tensorflow as tf
import tensorflow_addons as tfa

from ..base_runner import BaseRunner


class LookaheadAlgorithmRunner(BaseRunner):
    """A version of the BaseRunner that uses the Lookahead optimizer. """
    def _get_optimizer(self):
        hp = {k: v for k, v in self.hyper_parameters.items() if k != 'fast_step_size'}
        if 'fast_step_size' in self.hyper_parameters:
            fast_step = tf.keras.optimizers.SGD(learning_rate=self.hyper_parameters['fast_step_size'])
        else:
            fast_step = tf.keras.optimizers.SGD()

        return tfa.optimizers.Lookahead(fast_step, **hp)
