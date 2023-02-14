from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DataXy


@dataclass
class MixedStratifiedKFold:
    """
    This class wraps a `BaseCrossValidator` object, but accepts 2 datasets and
    returns folds over the two datasets.

    Its initializer accepts two `DataXy` instances and the probability of mixing
    data from the two datasets -- p=0 -> only the second dataset is used, p=1 ->
    only the first dataset is used.

    Its `split` yield indices as usual, but referred to the fully concatenated
    dataset retrievable by `get_full_data`
    """

    data_a: DataXy
    data_b: DataXy
    p: float
    base_splitter: object
    random_state: object = None

    def __post_init__(self):
        self.full_data_ = DataXy(
            pd.concat([self.data_a.X, self.data_b.X], axis=0),
            pd.concat([self.data_a.y, self.data_b.y], axis=0),
        )

    def get_n_splits(self, *args):
        return self.base_splitter.get_n_splits(*args)

    @property
    def shuffle(self):
        return self.base_splitter.shuffle

    def get_full_data(self):
        return self.full_data_

    def _stratified_mix_susbsample(self, arr_a, arr_b, y_a_ratios, y_b_ratios):
        N = min(arr_a.shape[0], arr_b.shape[0])

        N_a = int(self.p * N)
        N_b = N - N_a
        y_a_ratios = _n(y_a_ratios[arr_a])
        y_b_ratios = _n(y_b_ratios[arr_b])

        arr_a = self.random_state.choice(
            arr_a, size=N_a, replace=False, p=y_a_ratios, shuffle=False
        )
        arr_b = self.random_state.choice(
            arr_b, size=N_b, replace=False, p=y_b_ratios, shuffle=False
        )
        arr_b += arr_a.shape[0]
        return np.concatenate([arr_a, arr_b])

    def _init_split(self):
        if isinstance(self.random_state, int):
            self.random_state = np.random.default_rng(self.random_state)
        y_a = self.data_a.get_classes()
        y_b = self.data_b.get_classes()
        y_a_probs = _get_probs_from_class_ratio(y_a)
        y_b_probs = _get_probs_from_class_ratio(y_b)

        splitter_a = self.base_splitter.split(self.data_a.X, y_a)
        splitter_b = self.base_splitter.split(self.data_b.X, y_b)

        k1 = self.base_splitter.get_n_splits(self.data_a.X, y_a)
        k2 = self.base_splitter.get_n_splits(self.data_b.X, y_b)
        assert (
            k1 == k2
        ), "Error, the dataset objects received in conjunction with the cross-validator received generate two different number of folders"

        return k1, splitter_a, splitter_b, y_a_probs, y_b_probs

    def custom_split(self):
        """
        This returns three arrays:
            1 for train
            2. indices for test on dataset A
            2. indices for test on dataset B
        """
        k1, splitter_a, splitter_b, y_a_props, y_b_probs = self._init_split()

        for iteration in range(k1):
            train_a, test_a = next(splitter_a)
            train_b, test_b = next(splitter_b)

            # subsampling while keeping the proportion of the classes inferred
            train = self._stratified_mix_susbsample(
                train_a, train_b, y_a_props, y_b_probs
            )

            yield train, test_a, test_b + self.data_a.n_samples

    def split(self, X=None, y=None, groups=None):
        k1, splitter_a, splitter_b, y_a_probs, y_b_probs = self._init_split()

        for iteration in range(k1):
            train_a, test_a = next(splitter_a)
            train_b, test_b = next(splitter_b)

            # subsampling while keeping the proportion of the classes inferred
            train = self._stratified_mix_susbsample(
                train_a, train_b, y_a_probs, y_b_probs
            )
            test = self._stratified_mix_susbsample(
                test_a, test_b, y_a_probs, y_b_probs
            )

            yield train, test


def _get_probs_from_class_ratio(arr):
    """returns an array where each value is substituted by the ratio between that
    value and the total number of elements in `arr`"""
    out = arr.copy().astype(np.float)
    vals, count = np.unique(arr, return_counts=True)
    tot = count.sum()
    for i in range(vals.shape[0]):
        out[out == vals[i]] = count[i] / tot
    return out


def _n(arr):
    return arr / arr.sum()
