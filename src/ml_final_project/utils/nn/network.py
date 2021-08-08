import tensorflow as tf


def network(input_size, inner_dim, output_dim, num_inner_layers=4):
    x = nn_input = tf.keras.layers.Input((input_size,))
    for k in [2**i for i in range(0, num_inner_layers)][::-1]:
        x = tf.keras.layers.Dense(k*inner_dim, activation='relu')(x)

        if k > 1:
            x = tf.keras.layers.Dropout(0.7)(x)

    x = tf.keras.layers.Dense(output_dim, activation='relu')(x)
    nn_output = tf.keras.layers.Softmax()(x)

    return tf.keras.Model(inputs=[nn_input], outputs=[nn_output])
