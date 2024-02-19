from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from .data import DataXy


@dataclass
class AugmentedStratifiedKFold(BaseCrossValidator):
    """
    This class wraps a `BaseCrossValidator` object, but accepts 2 datasets and
    returns folds over the two datasets.

    Its initializer accepts two `DataXy` instances and the probability of
    addig
    data from the second datasets -- p=0 -> only the first dataset is used, p=1 ->
    both the first ad secod datasets are entirely used.

    Its `split` yield indices as usual, but referred to the fully concatenated
    dataset retrievable by `get_full_data`
    """

    data_a: DataXy
    data_b: DataXy
    p: float
    base_splitter: object
    random_state: object = None
    complementary_ratios: bool = False

    def __post_init__(self):
        self.random_state = np.random.default_rng(self.random_state)
        self.set_p(self.p)

    def set_complementary_ratios(self, newval):
        self.complementary_ratios = newval
        self.set_p(self.p)

    def set_p(self, p):
        self.p = p
        if self.complementary_ratios:
            self.q = 1 - self.p
        else:
            self.q = 1.0

    def get_n_splits(self, *args, **kwargs):
        return self.base_splitter.get_n_splits(*args, **kwargs)

    def swap(self):
        self.data_a, self.data_b = self.data_b, self.data_a

    @property
    def shuffle(self):
        return self.base_splitter.shuffle

    def get_full_data(self):
        """
        Returns a DataXy object containing the full data from both datasets.
        The name of the returned object is a combination of the names of the
        two datasets, including the p and q values. Note that, however, the
        returned object is not a mix but a full sum of the original objects. The
        returned object is inteded to be used for being indexed by the indices of
        `custom_split()`
        """
        return DataXy(
            pd.concat(
                [self.data_a._X_backup, self.data_b._X_backup], axis=0
            ).reset_index(drop=True),
            pd.concat(
                [self.data_a._y_backup, self.data_b._y_backup], axis=0
            ).reset_index(drop=True),
            name=f"{self.q}⨉{self.data_a.name}+{self.p}⨉{self.data_b.name}",
            random_state=self.data_a.random_state,
        )

    def get_augmented_data_size(self):
        return round(self.q * self.data_a.n_samples) + round(
            self.p * self.data_b.n_samples
        )

    def _stratified_augmented_susbsample(self, arr_a, arr_b, y_a_ratios, y_b_ratios):
        """
        This method takes two arrays (`arr_a` and `arr_b`) and returns a
        single array containing `part of arr_a` + part of `arr_b`, according to
        `self.p`.
        The sub-sampling happens according to the probability
        distribution in `y_a_ratios` and `y_b_ratios`.

        Input and output arrays are expected to contain indices. The output
        array contains indices related to `self.get_full_data()`
        """

        assert 0 <= self.p <= 1
        assert 0 <= self.q <= 1

        n_a = round(self.q * arr_a.shape[0])
        y_a_ratios = _n(y_a_ratios[arr_a])
        n_b = round(self.p * arr_b.shape[0])
        y_b_ratios = _n(y_b_ratios[arr_b])

        arr_a = self.random_state.choice(
            arr_a, size=n_a, replace=False, p=y_a_ratios, shuffle=False
        )
        arr_b = self.random_state.choice(
            arr_b, size=n_b, replace=False, p=y_b_ratios, shuffle=False
        )
        arr_b += arr_a.shape[0]  # the indices are all incremented!
        return np.concatenate([arr_a, arr_b])

    def _init_split(self):
        y_a = self.data_a.get_classes()
        y_b = self.data_b.get_classes()
        y_a_probs = self.data_a.get_y_probs()
        y_b_probs = self.data_b.get_y_probs()

        splitter_a = self.base_splitter.split(self.data_a.X, y_a)
        splitter_b = self.base_splitter.split(self.data_b.X, y_b)

        k1 = self.base_splitter.get_n_splits(self.data_a.X, y_a)
        k2 = self.base_splitter.get_n_splits(self.data_b.X, y_b)
        assert (
            k1 == k2
        ), "Error, the dataset objects received in conjunction with the cross-validator received generate two different number of folders"

        return k1, splitter_a, splitter_b, y_a_probs, y_b_probs

    def split(self, *args, **kwargs):
        """
        This returns two arrays:
            1 for train
            2. indices for test on both datasets A and B
        """
        for train, test_a, test_b in self.custom_split():
            yield train, np.concatenate([test_a, test_b])

    def custom_split(self):
        """
        This returns three arrays:
            1 for train
            2. indices for test on dataset A
            3. indices for test on dataset B
        """
        k1, splitter_a, splitter_b, y_a_probs, y_b_probs = self._init_split()

        for iteration in range(k1):
            train_a, test_a = next(splitter_a)
            train_b, test_b = next(splitter_b)

            # subsampling while keeping the proportion of the classes inferred
            train = self._stratified_augmented_susbsample(
                train_a, train_b, y_a_probs, y_b_probs
            )

            yield train, test_a, test_b + test_a.shape[0]


def _n(arr):
    return arr / arr.sum()
