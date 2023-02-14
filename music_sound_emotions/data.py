from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import settings as S


@dataclass
class DataXy:
    X: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        assert (
            self.X.shape[0] == self.y.shape[0]
        ), f"Error, `X` and `y` should have the same number of samples, but received {self.X.shape[0]} and {self.y.shape[0]}"
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self._y_backup = self.y.copy()

        # computing classes
        self.y_classes_ = np.zeros((self.y.shape[0],), dtype=np.int8)
        # val > 0, aro > 0 -> 1
        # val < 0, aro > 0 -> 2
        # TODO: fix for numpy arrays
        # TODO: fix 4.5
        self.y_classes_[self.y['AroMN'] > 4.5] = 1
        self.y_classes_[self.y['ValMN'] < 4.5] += 1
        # val < 0, aro < 0 -> 3
        # val > 0, aro < 0 -> 4
        self.y_classes_[self.y['AroMN'] < 4.5] = 3
        self.y_classes_[self.y['ValMN'] > 4.5] += 1

    def set_label(self, label: str):
        self.y = self.y_backup[label]

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
    return X, y


def load_data_x(dirs, fname):
    out = []
    for dir in dirs:
        filepath = Path(dir) / fname
        out.append(pd.read_csv(filepath, sep=";"))
    out = pd.concat(out)
    out.rename(columns={"name": "ID"}, inplace=True)
    out["ID"] = out["ID"].str[1:-5].astype(str)
    _2 = out["ID"].str.endswith('_2')
    out.loc[_2, 'ID'] = out.loc[_2, 'ID'].str[:-2]
    return out


def load_iads_y(iads_extended_dir):
    dir = Path(iads_extended_dir)
    df = pd.read_excel(dir / "Sound Ratings.xlsx")
    df.rename(columns={"Sound ID": "ID"}, inplace=True)
    df['ID'] = df['ID'].astype(str)
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
