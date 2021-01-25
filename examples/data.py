from pathlib import Path
import numpy as np

"""
Read and process UCI data from repository format.

Examples:

    >>> wine_dir = UciDataDir(Path('uci') / 'wine')
    >>> wine_dir.data.shape
    >>> train, test = wine_dir.get_split(17)
    >>> train.shape
    >>> test.shape
    >>> train_x, train_y = wine_dir.features_target(train)
    >>> train_x.shape
    >>> train_y.shape
"""


class UciDataDir:

    def __init__(self, path):
        self.path = path

    @property
    def data(self):
        return np.loadtxt(self.path / 'data.txt')

    @property
    def index_features(self):
        return np.loadtxt(self.path / 'index_features.txt', dtype=int)

    @property
    def index_target(self):
        return np.loadtxt(self.path / 'index_target.txt', dtype=int)

    @property
    def num_splits(self):
        return np.loadtxt(self.path / 'n_splits.txt', dtype=int)

    def get_split_indexes(self, split):
        return (
            np.loadtxt(self.path / f'index_train_{split}.txt', dtype=int),
            np.loadtxt(self.path / f'index_test_{split}.txt', dtype=int))

    def get_split(self, split):
        train_indexes, test_indexes = self.get_split_indexes(split)
        return data[train_indexes], data[test_indexes]

    def features_target(self, data):
        return (
            data[:, self.index_features],
            data[:, self.index_target])


wine_dir = UciDataDir(Path('uci') / 'wine')
wine_dir.data.shape
train, test = wine_dir.get_split(17)
train.shape
test.shape
train_x, train_y = wine_dir.features_target(train)
train_x.shape
train_y.shape
