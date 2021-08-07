import keras

from ..base_runner import BaseRunner


class MomentumAlgorithmRunner(BaseRunner):
    def _get_optimizer(self):
        return keras.optimizers.SGD(**self.hyper_parameters)
