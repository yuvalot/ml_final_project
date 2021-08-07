import pandas as pd
from sklearn.model_selection import KFold

from ..evaluators.requested import requested_evaluator
from ..hyperparameters.hyperparameter_scanner import HyperParameterScanner


class DatasetSimulator:
    def __init__(self, X, y, output_dim, algorithm_runner_class, scan_space):
        self.X = X
        self.y = y
        self.output_dim = output_dim
        self.algorithm_runner_class = algorithm_runner_class
        self.scan_space = scan_space

    def evaluate_batch(self, X_train, X_test, y_train, y_test):
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
        kf = KFold(n_splits=10, random_state=None, shuffle=False)
        results = []

        for i, (train_index, test_index) in enumerate(kf.split(self.X)):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            evaluation, training_time, hp = self.evaluate_batch(X_train, X_test, y_train, y_test)
            results.append({'Cross Validation [1-10]': 1 + i, 'Hyper-Parameters Values': hp, **evaluation,
                            'Training Time': training_time})

        return pd.DataFrame(results)
