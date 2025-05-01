import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def split_by_label(data, m, n):
    split = data[data['Label'].between(m, n)]

    return split


def accumulate_data(data1, data2):
    combined_data = pd.concat([data1, data2], axis=0, ignore_index=True)

    return combined_data


def split_train_test(data, test_size, random_state=11):
    X = data['Direction_Sequence']
    y = data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    return train, test


def split_data_label(data):
    return data['Direction_Sequence'], data['Label']


def to_input(data, MAX_LABEL):
    seq, label = split_data_label(data)

    seq = np.stack(seq.values)
    seq = seq[..., np.newaxis]

    label = label.values
    label = to_categorical(label, num_classes = MAX_LABEL)

    return seq, label