from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from . import settings as S


@dataclass
class DataXy:
    X: np.ndarray
    y: np.ndarray
    min_class_cardinality: int = S.N_SPLITS * 2

    def __post_init__(self):
        assert (
            self.X.shape[0] == self.y.shape[0]
        ), f"Error, `X` and `y` should have the same number of samples, but received {self.X.shape[0]} and {self.y.shape[0]}"
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self._y_backup = self.y.copy()

        # computing classes
        y_ = self.y[["AroMN", "ValMN"]] / 1
        self.y_classes_ = _cluster(y_, min_cardinality=self.min_class_cardinality)

    def set_label(self, label: str):
        self.y = self._y_backup[label]

    def get_classes(self):
        return self.y_classes_


def load_data():
    iads_x = load_data_x(S.IADS_DIR, S.FEATURE_FILE)
    iads_y = load_iads_y(S.IADSE_DIR)
    iads = DataXy(*_merge(iads_x, iads_y))

    pmemo_x = load_data_x(S.PMEMO_DIR, S.FEATURE_FILE)
    pmemo_y = load_pmemo_y(S.PMEMO_DIR[0])
    pmemo = DataXy(*_merge(pmemo_x, pmemo_y))

    return iads, pmemo


def _merge(X, y):
    df = X.merge(y, on="ID")
    X = df[X.columns].drop(columns=["ID"])
    y = df[y.columns].drop(columns=["ID"])
    y /= 9
    return X, y


def load_data_x(dirs, fname):
    out = []
    for dir in dirs:
        filepath = Path(dir) / fname
        out.append(pd.read_csv(filepath, sep=";"))
    out = pd.concat(out)
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
    return df[["ID", "AroMN", "AroSD", "ValMN", "ValSD"]]


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


def _cluster(X, metric="euclidean", linkage="ward", min_cardinality=5):

    print("  [Clustering]")
    assert min_cardinality < X.shape[0]

    K = X.shape[0]
    # define the agglomerative clustering model
    model = AgglomerativeClustering(
        # in sklearn 1.2 affinity -> metric
        n_clusters=K // min_cardinality,
        affinity=metric,
        distance_threshold=None,
        linkage=linkage,
    )

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
    print("  [Ended]")

    return labels
