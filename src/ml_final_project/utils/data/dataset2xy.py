from tensorflow.keras.utils import to_categorical


def dataset2Xy(dataset):
    """Convert a dataset (pd.DataFrame) to X, y and output_dim
        where X is the features, y is the labels (one-hot vectors),
        and output_dim is the number of labels overall.

        Args:
          dataset: A pandas dataframe that is composed of features
            columns and the class column which is the last one.

        Returns:
          tuple (X, y, output_dim) where X is the features matrix;
            y is the one-hot label matrix; and output_dim is the
            amount of different labels in y.
    """
    output_col = dataset.columns[-1]
    output_dim = len(dataset[output_col].value_counts())
    X = dataset.drop(columns=[output_col]).to_numpy()
    y = to_categorical(dataset[output_col].to_numpy(), num_classes=output_dim)
    return X, y, output_dim
