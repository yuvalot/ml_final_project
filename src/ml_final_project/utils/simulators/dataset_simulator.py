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

        return runner.evaluate(X_train, X_test, y_train, y_test, self.output_dim)
