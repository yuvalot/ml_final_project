def default_evaluator(model, X_test, y_test):
    """A simple evaluator that takes in a model,
        and a test set, and returns the loss.

        Args:
          model: The model to evaluate.
          X_test: The features matrix of the test set.
          y_test: The one-hot labels matrix of the test set.

        Returns:
          The loss on the test set.
    """
    return model.evaluate(X_test, y_test, verbose=0)[0]
