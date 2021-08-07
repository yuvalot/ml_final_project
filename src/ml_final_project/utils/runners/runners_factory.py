from .specific.improved_lookahead import ImprovedLookaheadAlgorithmRunner
from .specific.lookahead import LookaheadAlgorithmRunner
from .specific.momentum import MomentumAlgorithmRunner

available_runners = {
    'momentum': MomentumAlgorithmRunner,
    'lookahead': LookaheadAlgorithmRunner,
    'improved_lookahead': ImprovedLookaheadAlgorithmRunner
}


def get_runner(optimizer: str):
    if optimizer in available_runners:
        return available_runners[optimizer]
    else:
        raise NotImplementedError()
