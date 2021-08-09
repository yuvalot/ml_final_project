import tensorflow as tf


def network(input_size, inner_dim, output_dim, num_inner_layers=4):
    """A generic neural network creator. The architecture is:
        1) Input layer.
        2) Inner num_inner_layers dense layers, of sizes
            inner_dim * (2 ** (num_inner_layers-1)), ... , inner_dim * (2 ** (0))
        3) Dense layer of size output_dim.
        4) A Softmax layer.

        Args:
          input_size: The size of the input.
          inner_dim: controls the size (width) of the inner layers
              (by the way described above).
          output_dim: The size of the output.
          num_inner_layers: controls the number of the inner layers
              (by the way described above).

        Returns:
          A keras Model as described above.
    """
    x = nn_input = tf.keras.layers.Input((input_size,))
    for k in [2**i for i in range(0, num_inner_layers)][::-1]:
        x = tf.keras.layers.Dense(k*inner_dim, activation='relu')(x)

        if k > 1:
            x = tf.keras.layers.Dropout(0.7)(x)

    x = tf.keras.layers.Dense(output_dim, activation='relu')(x)
    nn_output = tf.keras.layers.Softmax()(x)

    return tf.keras.Model(inputs=[nn_input], outputs=[nn_output])
