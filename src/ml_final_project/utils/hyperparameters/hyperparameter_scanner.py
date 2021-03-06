import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold


class HyperParameterScanner:
    """A simple helper class that scans for hyper parameters
        for the algorithm. It wraps the bayes_opt module, and
        use it to search the best set of hyper parameters. It
        does 3 fold cross validation to evaluate a single
        configuration of hyper parameters.

        Attributes:
            X - features matrix.
            y - one-hot labels matrix.
            output_dim - the amount of labels.
            algorithm_runner_class - the algorithm runner class
                (used to create an algorithm runner). See utils/runners.
            scan_space - the scan space of hyper parameters to scan.
    """
    def __init__(self, X, y, output_dim, algorithm_runner_class, scan_space):
        self.X = X
        self.y = y
        self.output_dim = output_dim
        self.algorithm_runner_class = algorithm_runner_class
        self.scan_space = scan_space

    def optimize(self):
        """Finds the optimal set of hyper parameters.

            Returns:
                A dict of the hyper parameters.
        """
        def evaluate(**kwargs):
            return -self.evaluate(kwargs)

        optimizer = BayesianOptimization(
            f=evaluate,
            pbounds=self.scan_space
        )

        optimizer.maximize(
            init_points=5,
            n_iter=45,
        )

        return optimizer.max['params']

    def evaluate(self, hyper_parameters):
        """Evaluate a single set of hyper parameters using
            3-fold cross validation.

            Returns:
                The evaluation.
        """
        runner = self.algorithm_runner_class(hyper_parameters)
        kf = StratifiedKFold(n_splits=3, random_state=None, shuffle=False)
        agg = 0

        for train_index, test_index in kf.split(self.X, np.argmax(self.y, axis=1)):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            agg += runner.evaluate(X_train, X_test, y_train, y_test, self.output_dim)[0]

        return agg / 3.0
