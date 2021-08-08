# These are the hyper-parameters scanned for each algorithm, and its scan space
scan_spaces = {
    'momentum': {'learning_rate': (1e-6, 1e-1), 'momentum': (0, 1)},
    'lookahead': {'fast_step_size': (1e-6, 1e-1), 'slow_step_size': (1e-6, 1e-1)},
    'improved_lookahead': {'fast_step_size': (1e-6, 1e-1), 'slow_step_size': (1e-6, 1e-1), 'momentum': (0, 1)}
}

# These are configuration for the neural network
network_conf = {
    'inner_dim': 64,
    'num_inner_layers': 4,
    'batch_size': 32,
    'epochs': 30,
    'val': 1
}
