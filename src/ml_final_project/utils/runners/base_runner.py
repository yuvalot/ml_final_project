from datetime import datetime

import tensorflow as tf

from ..evaluators.default import default_evaluator
from ..nn.network import network
from ...consts import network_conf


class BaseRunner:
    def __init__(self, hyper_parameters):
        self.hyper_parameters = hyper_parameters

    def _get_optimizer(self):
        raise NotImplementedError()

    def evaluate(self, X_train, X_test, y_train, y_test, output_dim, evaluator=default_evaluator):
        model = network(input_size=X_train.shape[1], inner_dim=network_conf['inner_dim'], output_dim=output_dim, num_inner_layers=network_conf['num_inner_layers'])
        model.compile(optimizer=self._get_optimizer(), loss="categorical_crossentropy", metrics=["accuracy"])
        if network_conf['val'] == 1:
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            val = {'callbacks': [es], 'validation_split': .2}
        else:
            val = {}
        start_training_time = datetime.now()
        model.fit(X_train, y_train, epochs=network_conf['epochs'], verbose=0, batch_size=network_conf['batch_size'], **val)
        end_training_time = datetime.now()
        return evaluator(model, X_test, y_test), (end_training_time - start_training_time).total_seconds()
