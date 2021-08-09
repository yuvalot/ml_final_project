import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..evaluators.requested import requested_evaluator
from ..hyperparameters.hyperparameter_scanner import HyperParameterScanner


class DatasetSimulator:
    """A simple helper class that uses the other class in this
        project to run a full simulation and evaluation of the
        algorithm on the dataset. It does external 10-fold cross
        validation and internal 3-fold cross validation for
        hyper parameters tuning.

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

    def evaluate_batch(self, X_train, X_test, y_train, y_test):
        """evaluates the algorithm using a given train/test sets.

            Args:
              X_train: The features of the training set.
              y_train: The labels of the training set.
              X_test: The features of the test set.
              y_test: The labels of the test set.

            Returns:
              The result of the evaluator, the training time, and the optimal hyper parameters
        """
        scanner = HyperParameterScanner(
            X=X_train,
            y=y_train,
            output_dim=self.output_dim,
            algorithm_runner_class=self.algorithm_runner_class,
            scan_space=self.scan_space
        )

        optimal_hyper_parameters = scanner.optimize()
        runner = self.algorithm_runner_class(optimal_hyper_parameters)
        evaluation, training_time = runner.evaluate(X_train, X_test, y_train, y_test, self.output_dim,
                                                    evaluator=requested_evaluator)
        return evaluation, training_time, optimal_hyper_parameters

    def evaluate(self):
        """evaluates the algorithm using a given train/test sets.

            Returns:
              The results of the evaluators, the training times,
                  and the optimal hyper parameters, collected in a
                  pandas Dataframe.
        """
        kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
        results = []

        for i, (train_index, test_index) in enumerate(kf.split(self.X, np.argmax(self.y, axis=1))):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            evaluation, training_time, hp = self.evaluate_batch(X_train, X_test, y_train, y_test)
            results.append({'Cross Validation [1-10]': 1 + i, 'Hyper-Parameters Values': hp, **evaluation,
                            'Training Time': training_time})

        return pd.DataFrame(results)
