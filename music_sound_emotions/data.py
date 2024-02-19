from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from . import settings as S
from .settings import tlog


@dataclass
class DataXy:
    X: np.ndarray
    y: np.ndarray
    min_class_cardinality: int = S.N_SPLITS**2
    n_clusters: int = None
    name: str = ""
    music_ids: list = None
    random_state: Optional[int] = None

    def __post_init__(self):
        assert (
            self.X.shape[0] == self.y.shape[0]
        ), f"Error, `X` and `y` should have the same number of samples, but received {self.X.shape[0]} and {self.y.shape[0]}"
        if self.random_state is None or isinstance(self.random_state, int):
            self.random_state = np.random.default_rng(self.random_state)
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self._X_backup = self.X.copy()
        self._y_backup = self.y.copy()
        self.current_label_ = None

    def init_classes(self):
        """Computes the classes and the probabilities of the classes. The
        classes are computed as clusters of the y values (valence and arousal)
        and the probabilities as the ratio between the number of elements in
        each class and the total number of elements. These classes and
        probabilities are only used for stratified k-fold cross-validation."""
        # computing classes
        y_ = self._y_backup[["AroMN", "ValMN"]]
        self.y_classes_ = _cluster(
            y_,
            min_cardinality=self.min_class_cardinality,
            n_clusters=self.n_clusters,
        )

        # computing probs
        y_probs_ = self.y_classes_.copy().astype(np.float)
        vals, count = np.unique(self.y_classes_, return_counts=True)
        for i in range(vals.shape[0]):
            y_probs_[y_probs_ == vals[i]] = count[i]
        self.y_probs_ = y_probs_ / y_probs_.sum()

    def set_label(self, label: str):
        """Set the label of the dataset to `label`. `label` is the column used
        as target (the y)"""
        self.current_label_ = label
        if label is None:
            self.unset_label()
        else:
            self.y = self._y_backup[label]
        return self

    def unset_label(self):
        self.current_label_ = None
        self.y = self._y_backup
        return self

    def get_classes(self):
        if not hasattr(self, "y_classes_"):
            self.init_classes()
        return self.y_classes_

    def get_y_probs(self):
        """returns an array where each value is substituted by the ratio between that
        value and the total number of elements in `data.get_classes`"""
        return self.y_probs_

    def remove_music_ids(self):
        """Removes the music ids from the dataset if `music_ids` is not None. Works in-place!"""
        if self.music_ids is not None:
            self.X = self.X.drop(index=self.music_ids)
            self.y = self.y.drop(index=self.music_ids)
            current_label = self.current_label_
            self.__post_init__()
            self.name += "-nomusic"
            self.current_label_ = current_label
        else:
            pass

    def truncate(self, n_samples):
        """Truncates the dataset to `n_samples` samples"""

        if not hasattr(self, "y_probs_"):
            self.init_classes()
        chosen_idx = np.random.choice(
            np.arange(self.X.shape[0]),
            n_samples,
            replace=False,
            p=self.y_probs_,
        )
        self.X = self.X.iloc[chosen_idx]
        self.y = self.y.iloc[chosen_idx]

        current_label = self.current_label_
        self.__post_init__()
        self.current_label_ = current_label


def load_data(normalize=True):
    """
    If `normalize` is True, than IADS-E is normalized so that 1 -> -1 and 9 -> 1,
    while pmemo is normalized so that 0 -> -1 and 1 -> 1
    """
    iads_x = load_data_x(S.IADS_DIR, S.FEATURE_FILE)
    iads_y, iads_category = load_iads_y(S.IADSE_DIR)
    if normalize:
        iads_y[["ValMN", "AroMN"]] = (iads_y[["ValMN", "AroMN"]] - 5) / 4
    iads_x, iads_y, ids = _merge(iads_x, iads_y, return_ids_col=True)
    music_ids = ids[
        ids.isin(iads_category["ID"][iads_category["Category"] == "Music"])
    ].index
    iads = DataXy(iads_x, iads_y, music_ids=music_ids, name="IADS-E", random_state=1970)

    pmemo_x = load_data_x(S.PMEMO_DIR, S.FEATURE_FILE)
    pmemo_y = load_pmemo_y(S.PMEMO_DIR[0])
    if normalize:
        pmemo_y[["ValMN", "AroMN"]] = pmemo_y[["ValMN", "AroMN"]] * 2 - 1
    pmemo = DataXy(*_merge(pmemo_x, pmemo_y), name="PMEmo", random_state=1996)

    return iads, pmemo


def _merge(X, y, return_ids_col=False):
    # drop_duplicates is due to the fact that I have extracted each file in PMEmo twice
    df = X.merge(y, on="ID").drop_duplicates()
    ids = df["ID"]
    X = df[X.columns].drop(columns=["ID"])
    y = df[y.columns].drop(columns=["ID"])
    if return_ids_col:
        return X, y, ids
    else:
        return X, y


def load_data_x(dirs, fname):
    out = []
    for dir in dirs:
        filepath = Path(dir) / fname
        out.append(pd.read_csv(filepath, sep=";"))
    out = pd.concat(out)
    del out["frameTime"]
    out.rename(columns={"name": "ID"}, inplace=True)
    out["ID"] = out["ID"].str[1:-5].astype(str)
    _2 = out["ID"].str.endswith("_2")
    out.loc[_2, "ID"] = out.loc[_2, "ID"].str[:-2]
    return out


def load_iads_y(iads_extended_dir):
    dir = Path(iads_extended_dir)
    df = pd.read_excel(dir / "Sound Ratings.xlsx")
    df.rename(columns={"Sound ID": "ID"}, inplace=True)
    df["ID"] = df["ID"].astype(str)
    return df[["ID", "AroMN", "AroSD", "ValMN", "ValSD"]], df[["ID", "Category"]]


def load_pmemo_y(pmemo_dir):
    dir = Path(pmemo_dir)
    means = pd.read_csv(dir / "annotations" / "static_annotations.csv")
    std = pd.read_csv(dir / "annotations" / "static_annotations_std.csv")
    metadata = pd.read_csv(dir / "metadata.csv")
    means = metadata.merge(means, on="musicId")
    std = metadata.merge(std, on="musicId")
    df = means.merge(std, on="musicId")
    df.rename(
        columns={
            "fileName_x": "ID",
            "Arousal(mean)": "AroMN",
            "Arousal(std)": "AroSD",
            "Valence(mean)": "ValMN",
            "Valence(std)": "ValSD",
        },
        inplace=True,
    )
    df["ID"] = df["ID"].str[:-4]
    return df[["ID", "AroMN", "AroSD", "ValMN", "ValSD"]]


def _cluster(X, metric="euclidean", linkage="ward", min_cardinality=5, n_clusters=None):
    """
    Returns the clusters of X so that the largest umber of clusters is returned,
    while keeping the minimum caridnality equal to `min_cardinality`
    """

    tlog("  [Clustering]")
    assert min_cardinality or n_clusters
    if min_cardinality:
        assert min_cardinality < X.shape[0]

    K = X.shape[0]
    # define the agglomerative clustering model
    model = AgglomerativeClustering(
        # in sklearn 1.2 affinity -> metric
        n_clusters=K // min_cardinality if min_cardinality else n_clusters,
        affinity=metric,
        distance_threshold=None,
        linkage=linkage,
    )

    if n_clusters:
        return model.fit_predict(X)

    # fit the model to the data
    model.fit(X)

    # get the cluster labels for each data point
    labels = model.labels_

    # get the number of clusters
    t = np.unique(labels, return_counts=True)[1].min()

    # stop the procedure if all clusters have cardinality >= x
    while t < min_cardinality:
        # merge the two closest clusters
        model.n_clusters -= 1
        labels = model.fit_predict(X)
        t = np.unique(labels, return_counts=True)[1].min()
    tlog("  [Ended]")

    return labels
