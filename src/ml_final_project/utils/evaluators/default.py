def default_evaluator(model, X_test, y_test):
    return model.evaluate(X_test, y_test, verbose=0)[0]
