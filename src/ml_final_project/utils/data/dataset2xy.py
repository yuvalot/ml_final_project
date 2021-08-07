from tensorflow.keras.utils import to_categorical


def dataset2Xy(dataset):
    output_col = dataset.columns[-1]
    output_dim = len(dataset[output_col].value_counts())
    X = dataset.drop(columns=[output_col]).to_numpy()
    y = to_categorical(dataset[output_col].to_numpy(), num_classes=output_dim)
    return X, y, output_dim
