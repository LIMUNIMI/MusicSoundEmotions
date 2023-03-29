from dataclasses import dataclass

from sklearn.model_selection import BaseCrossValidator
import numpy as np
import pandas as pd

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

    def __post_init__(self):
        self.full_data_ = DataXy(
            pd.concat([self.data_a.X, self.data_b.X], axis=0).reset_index(drop=True),
            pd.concat([self.data_a.y, self.data_b.y], axis=0).reset_index(drop=True),
            name=f"{self.data_a.name}+{self.p}⨉{self.data_b.name}"
        )
        self.random_state = np.random.default_rng(self.random_state)

    def get_n_splits(self, *args, **kwargs):
        return self.base_splitter.get_n_splits(*args, **kwargs)

    def swap(self):
        """Swap the two datasets"""
        self.data_a, self.data_b = self.data_b, self.data_a

    @property
    def shuffle(self):
        return self.base_splitter.shuffle

    def get_full_data(self):
        return self.full_data_

    def get_augmented_data(self, **kwargs):
        idx_a = np.arange(self.data_a.n_samples)
        idx_b = np.arange(self.data_b.n_samples)
        b_probs = self.data_b.get_y_probs()
        augmented_idx = self._stratified_augmented_susbsample(idx_a, idx_b, b_probs)
        return DataXy(
            self.full_data_.X.loc[augmented_idx],
            self.full_data_.y.loc[augmented_idx],
            **kwargs,
        )

    def _stratified_augmented_susbsample(self, arr_a, arr_b, y_b_ratios):
        """
        This method takes two arrays (`arr_a` and `arr_b`) and returns a
        single array containing `arr_a` + part of `arr_b`, according to
        `self.p`.
        The sub-sampling happens according to the probability
        distribution in `y_b_ratios`.

        Input and output arrays are expected to contain indices. The output
        array contains indices related to `self.full_data_`
        """

        assert 0 <= self.p <= 1
        N_b = round(self.p * arr_b.shape[0])
        y_b_ratios = _n(y_b_ratios[arr_b])

        arr_b = self.random_state.choice(
            arr_b, size=N_b, replace=False, p=y_b_ratios, shuffle=False
        )
        arr_b += arr_a.shape[0]  # the indices are all incremented!
        return np.concatenate([arr_a, arr_b])

    def _init_split(self):
        y_a = self.data_a.get_classes()
        y_b = self.data_b.get_classes()
        # y_a_probs = self.data_a.get_y_probs()
        y_b_probs = self.data_b.get_y_probs()

        splitter_a = self.base_splitter.split(self.data_a.X, y_a)
        splitter_b = self.base_splitter.split(self.data_b.X, y_b)

        k1 = self.base_splitter.get_n_splits(self.data_a.X, y_a)
        k2 = self.base_splitter.get_n_splits(self.data_b.X, y_b)
        assert (
            k1 == k2
        ), "Error, the dataset objects received in conjunction with the cross-validator received generate two different number of folders"

        return k1, splitter_a, splitter_b, y_b_probs

    def split(self, *args, **kwargs):
        """
        This returns two arrays:
            1 for train
            2. indices for test on both datasets A and B
        """
        k1, splitter_a, splitter_b, y_b_probs = self._init_split()

        for iteration in range(k1):
            train_a, test_a = next(splitter_a)
            train_b, test_b = next(splitter_b)

            # subsampling while keeping the proportion of the classes inferred
            train = self._stratified_augmented_susbsample(train_a, train_b, y_b_probs)

            yield train, np.concatenate([test_a, test_b + self.data_a.n_samples])

    def custom_split(self):
        """
        This returns three arrays:
            1 for train
            2. indices for test on dataset A
            3. indices for test on dataset B
        """
        k1, splitter_a, splitter_b, y_b_probs = self._init_split()

        for iteration in range(k1):
            train_a, test_a = next(splitter_a)
            train_b, test_b = next(splitter_b)

            # subsampling while keeping the proportion of the classes inferred
            train = self._stratified_augmented_susbsample(train_a, train_b, y_b_probs)

            yield train, test_a, test_b + self.data_a.n_samples


def _n(arr):
    return arr / arr.sum()
