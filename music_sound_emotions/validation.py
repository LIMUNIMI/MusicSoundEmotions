import datetime
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy
from autosklearn.regression import AutoSklearnRegressor
from sklearn import metrics
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

from . import settings as S
from .settings import tlog
from .splits import AugmentedStratifiedKFold, DataXy
from .utils import logger, telegram_notify


def cross_validate(
    model,
    splitter: AugmentedStratifiedKFold,
    metrics: list,
    label: str,
):
    """
    Given a model, cross validates it on the augmented data while testing on
    the separated test folds
    """

    metrics_a = [[] for _ in metrics]
    metrics_b = [[] for _ in metrics]
    full_data = splitter.get_full_data()
    old_label = full_data.current_label_
    set_label(label, full_data)
    X, y = full_data.X.to_numpy(), full_data.y.to_numpy()
    splitter.proportional_test_folds = False
    for train, test_a, test_b in tqdm(splitter.custom_split(), desc="Cross-validating"):
        if isinstance(model, AutoSklearnRegressor):
            model.refit(X[train], y[train])
            model_ = model
        else:
            model_ = clone(model)
            model_.fit(X[train], y[train])

        y_a_cap = model_.predict(X[test_a])
        y_a_true = y[test_a]
        y_b_cap = model_.predict(X[test_b])
        y_b_true = y[test_b]

        for i, metric in enumerate(metrics):
            metrics_a[i].append(metric(y_a_true, y_a_cap))
            metrics_b[i].append(metric(y_b_true, y_b_cap))

    splitter.proportional_test_folds = True
    metrics_a = [(np.mean(m), confidence(m)) for m in metrics_a]
    metrics_b = [(np.mean(m), confidence(m)) for m in metrics_b]
    set_label(old_label, full_data)
    return metrics_a, metrics_b


def confidence(dist, conf_level=0.95):
    dist = np.asarray(dist)
    s = np.std(dist, ddof=1)
    n = dist.shape[0]
    df = n - 1
    t_val = scipy.stats.t.ppf((1 + conf_level) / 2, df)
    moe = t_val * s / np.sqrt(n)
    return moe


def set_label(label, *datasets):
    """set the target label of the datasets"""
    for dataset in datasets:
        dataset.set_label(label)


def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    return r2


@dataclass
class Main:
    order: tuple = ("IADS-E", "PMEmo")
    p: float = 0.5
    only_automl: bool = False
    noised: Optional[str] = None
    __complementary_ratios: bool = False
    __remove_iads_music = False

    def __post_init__(self):
        from .data import load_data

        tlog("Loading data")
        self.data1, self.data2 = load_data()
        if self.data1.name != self.order[0]:
            self.data1, self.data2 = self.data2, self.data1
        if self.order[0] == self.order[1]:
            self.data2 = self.data1
        if self.noised is not None:
            for d in (self.data1, self.data2):
                if self.noised == d.name:
                    # instead of shuffling the y data
                    # generate new y data between each column's min and max
                    rng = np.random.default_rng(2024)
                    d.y[:] = rng.uniform(d.y.min(), d.y.max(), d.y.shape)

        self.splitter = AugmentedStratifiedKFold(
            self.data1,
            self.data2,
            p=self.p,
            base_splitter=StratifiedKFold(
                n_splits=S.N_SPLITS, random_state=1983, shuffle=True
            ),
            # base_splitter=KFold(
            #     n_splits=S.N_SPLITS, random_state=1983, shuffle=True
            # ),
            random_state=1992,
            complementary_ratios=self.__complementary_ratios,
        )

    def _set_p(self, p):
        self.splitter.set_p(p)

    @property
    def complementary_ratios(self):
        return self.__complementary_ratios

    @complementary_ratios.setter
    def complementary_ratios(self, newval):
        """Warning! use this and remove_iads_music with caution as they're not guardanteed to reset to the previous state correctly if used together!"""
        if newval != self.__complementary_ratios:
            self.__complementary_ratios = newval
            self.splitter.set_complementary_ratios(newval)
            if newval:
                # was False, now it is True
                self.data1_complementary_backup_ = deepcopy(self.data1)
                self.data2_complementary_backup_ = deepcopy(self.data2)
                k_fixed = min(self.data1.X.shape[0], self.data2.X.shape[0])
                self.data1.truncate(k_fixed)
                self.data2.truncate(k_fixed)
            else:
                self.data1 = self.data1_complementary_backup_
                self.data2 = self.data2_complementary_backup_
            self.splitter.data_a = self.data1
            self.splitter.data_b = self.data2

    @property
    def remove_iads_music(self):
        return self.__remove_iads_music

    @remove_iads_music.setter
    def remove_iads_music(self, newval):
        """Warning! use this and complementary_ratios with caution as they're not guardanteed to reset to the previous state correctly if used together!"""
        if newval != self.__remove_iads_music:
            self.__remove_iads_music = newval
            if newval:
                # was False, now it is True
                self.data1_iads_music_backup_ = deepcopy(self.data1)
                self.data2_iads_music_backup_ = deepcopy(self.data2)
                for d in (self.data1, self.data2):
                    if d.name == "IADS-E":
                        d.remove_music_ids()
            else:
                # was True, now it is False
                self.data1 = self.data1_iads_music_backup_
                self.data2 = self.data2_iads_music_backup_
                for d in (self.data1, self.data2):
                    d.name = d.name.replace("-no-music", "")
            # in any case, apply the changes to the splitter
            self.splitter.data_a = self.data1
            self.splitter.data_b = self.data2

    def swap(self):
        self.data1, self.data2 = self.data2, self.data1
        self.splitter.swap()

    @logger.catch
    def tune_and_validate(self, label):
        from .models import get_tuners, save_and_get_best_model

        full_data = self.splitter.get_full_data()

        old_label = self.data1.current_label_
        set_label(label, self.data1, self.data2, full_data)  # , augmented_data)
        self.data1.init_classes()
        self.data2.init_classes()

        # compute l1 norm of data1.x - data2.x and data1.y - data2.y
        # then sort by the l1 norm of the two and print the 5 list values
        # m = min(self.data1.X.shape[0], self.data2.X.shape[0])
        # x_diff = np.abs(self.data1.X[:m].to_numpy() - self.data2.X[:m].to_numpy()).sum(
        #     axis=1
        # )
        # y_diff = np.abs(self.data1.y[:m].to_numpy() - self.data2.y[:m].to_numpy())
        # print(f"Most similar samples: {diff.nsmallest(5)}")
        for tuner in get_tuners(self.splitter, self.only_automl):
            tlog(f"Tuning {tuner['name']}")
            tlog._log_spaces += 4

            ttt = time.time()
            tuner["model"].fit(full_data.X.to_numpy(), full_data.y.to_numpy())
            # tuner["model"].fit(augmented_data.X.to_numpy(), augmented_data.y.to_numpy())
            print("Needed time for optimizing the model: ", time.time() - ttt)
            telegram_notify(f"{tuner['name']} done in {(time.time() - ttt)/60} minutes")
            # cross-validate best result
            tlog("Cross-validating best estimator")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{tuner['name']}_{self.splitter.p:.2f}-{timestamp}.pickle"
            ttt = time.time()
            data1_res, data2_res = cross_validate(
                save_and_get_best_model(tuner, filepath),
                self.splitter,
                [
                    r2_score,
                    lambda x, y: np.sqrt(np.mean((x - y) ** 2)),
                    metrics.mean_absolute_error,
                ],
                label,
            )
            print(
                "Needed time for cross-validating the best model: ", time.time() - ttt
            )

            tlog("___________________")
            tlog(f"Obtained metrics for {self.data1.name}")
            tlog("   r2, RMSE, MAE")
            for v, err in data1_res:
                tlog(f"{v:.2e} ± {err:.2e}")
            tlog("___________________")
            tlog(f"Obtained metrics for {self.data2.name}")
            tlog("   r2, RMSE, MAE")
            for v, err in data2_res:
                tlog(f"{v:.2e} ± {err:.2e}")
            tlog()
            tlog._log_spaces -= 4

        set_label(old_label, self.data1, self.data2, full_data)  # , augmented_data)


if __name__ == "__main__":
    import fire

    fire.Fire(Main)
    telegram_notify("Ended!")
