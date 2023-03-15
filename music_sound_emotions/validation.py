import pickle
from dataclasses import dataclass

import numpy as np
import scipy
from autosklearn.regression import AutoSklearnRegressor
from sklearn import metrics
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from . import settings as S
from .settings import tlog
from .splits import DataXy, AugmentedStratifiedKFold
from .utils import logger, telegram_notify


def cross_validate(
    model,
    data_a: DataXy,
    data_b: DataXy,
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
    for train, test_a, test_b in splitter.custom_split():
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
    for dataset in datasets:
        dataset.set_label(label)


@dataclass
class Main:
    order: tuple
    p: float = 0.5

    def __post_init__(self):
        from .data import load_data

        tlog("Loading data")
        self.data1, self.data2 = load_data()
        self.data1 = self.data1 if self.data1.name == self.order[0] else self.data2
        self.data2 = self.data2 if self.data2.name == self.order[1] else self.data1
        self.splitter = AugmentedStratifiedKFold(
            self.data1,
            self.data2,
            p=self.p,
            base_splitter=StratifiedKFold(
                n_splits=S.N_SPLITS, random_state=1983, shuffle=True
            ),
            random_state=1992,
        )

    def _set_p(self, p):
        self.splitter.p = p

    @logger.catch
    def tune_and_validate(self, label):
        from .models import get_best_model, get_tuners

        full_data = self.splitter.get_full_data()
        mixed_data = self.splitter.get_augmented_data(
            # n_clusters=S.N_SPLITS * 2,
            # min_class_cardinality=None
        )

        old_label = self.data1.current_label_
        set_label(label, self.data1, self.data2, full_data, mixed_data)

        for tuner in get_tuners(self.splitter.base_splitter):
            tlog(f"Tuning {tuner['name']}")
            tlog._log_spaces += 4
            # tuning hyperparameters
            if hasattr(tuner["model"], "set_y_classes"):
                tuner["model"].set_y_classes(
                    mixed_data.get_classes(), mixed_data.get_y_probs()
                )
            import time
            ttt = time.time()
            tuner["model"].fit(mixed_data.X.to_numpy(),
                               mixed_data.y.to_numpy())
            print("Time: ", time.time() - ttt)
            telegram_notify(f"{tuner['name']} done in {(time.time() - ttt)/60} minutes")
            # cross-validate best result
            tlog("Cross-validating best estimator")
            data1_res, data2_res = cross_validate(
                get_best_model(tuner),
                self.data1,
                self.data2,
                self.splitter,
                [
                    metrics.r2_score,
                    lambda x, y: metrics.mean_squared_error(x, y, squared=False),
                    metrics.mean_absolute_error,
                ],
                label,
            )

            pickle.dump(get_best_model(tuner), open(tuner["name"] + ".pkl", "wb"))

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

        set_label(old_label, self.data1, self.data2, full_data, mixed_data)


if __name__ == "__main__":
    import fire

    fire.Fire(Main)
    telegram_notify("Ended!")
