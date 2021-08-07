import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold


class HyperParameterScanner:
    def __init__(self, X, y, output_dim, algorithm_runner_class, scan_space):
        self.X = X
        self.y = y
        self.output_dim = output_dim
        self.algorithm_runner_class = algorithm_runner_class
        self.scan_space = scan_space

    def optimize(self):
        def evaluate(**kwargs):
            return -self.evaluate(kwargs)

        optimizer = BayesianOptimization(
            f=evaluate,
            pbounds=self.scan_space
        )

        optimizer.maximize(
            init_points=5,
            n_iter=1,
        )

        return optimizer.max['params']

    def evaluate(self, hyper_parameters):
        runner = self.algorithm_runner_class(hyper_parameters)
        kf = StratifiedKFold(n_splits=3, random_state=None, shuffle=False)
        agg = 0

        for train_index, test_index in kf.split(self.X, np.argmax(self.y, axis=1)):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            agg += runner.evaluate(X_train, X_test, y_train, y_test, self.output_dim)[0]

        return agg / 3.0
