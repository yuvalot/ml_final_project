import keras

from ..base_runner import BaseRunner
from ....optimizers.improved_lookahead import ImprovedLookahead


class ImprovedLookaheadAlgorithmRunner(BaseRunner):
    def _get_optimizer(self):
        return ImprovedLookahead(keras.optimizers.SGD(), **self.hyper_parameters)
