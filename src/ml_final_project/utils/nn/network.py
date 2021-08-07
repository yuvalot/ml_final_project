import keras


def network(input_size, inner_dim, output_dim):
    x = nn_input = keras.layers.Input((input_size,))
    for k in [8, 4, 2, 1]:
        x = keras.layers.Dense(k*inner_dim, activation='relu')(x)

        if k > 1:
            x = keras.layers.Dropout(0.7)(x)

    x = keras.layers.Dense(output_dim, activation='relu')(x)
    nn_output = keras.layers.Softmax()(x)

    return keras.Model(inputs=[nn_input], outputs=[nn_output])
