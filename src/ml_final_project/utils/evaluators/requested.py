def requested_evaluator(model, X_test, y_test):
    return {'accuraccy': model.evaluate(X_test, y_test, verbose=0)[0]}
