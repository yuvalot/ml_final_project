import pandas as pd
import numpy as np
from sklearn import preprocessing


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    for column in df.columns:
        if df[column].dtype == np.object:
            df[column] = preprocessing.LabelEncoder().fit_transform(df[column].fillna('0'))

    df[df.columns[-1]] = preprocessing.LabelEncoder().fit_transform(df[df.columns[-1]].fillna('0'))
    return df
