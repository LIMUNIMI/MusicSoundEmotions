import pickle
from dataclasses import dataclass

import numpy as np
import scipy
from autosklearn.regression import AutoSklearnRegressor
from sklearn import metrics
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from . import settings as S
from .splits import DataXy, MixedStratifiedKFold
from .settings import tlog


def cross_validate(
    model,
    data_a: DataXy,
    data_b: DataXy,
    splitter: MixedStratifiedKFold,
    metrics: list,
    label: str,
):
    """
    Given a model, cross validates it on the mixed data while testing on
    the separated test folds
    """

    metrics_a = [[] for _ in metrics]
    metrics_b = [[] for _ in metrics]
    full_data = splitter.get_full_data()
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
    p: float = 0.5

    def __post_init__(self):
        from .data import load_data

        tlog("Loading data")
        self.iads, self.pmemo = load_data()
        self.splitter = MixedStratifiedKFold(
            self.iads,
            self.pmemo,
            p=self.p,
            base_splitter=StratifiedKFold(
                n_splits=S.N_SPLITS, random_state=1983, shuffle=True
            ),
            random_state=1992,
        )

    def _set_p(self, p):
        self.splitter.p = p

    def tune_and_validate(self, label):

        from .models import get_best_model, get_tuners

        full_data = self.splitter.get_full_data()
        mixed_data = self.splitter.get_mixed_data(
            # n_clusters=S.N_SPLITS * 2,
            # min_class_cardinality=None
        )

        set_label(label, self.iads, self.pmemo, full_data, mixed_data)

        for tuner in get_tuners(self.splitter.base_splitter):
            tlog(f"Tuning {tuner['name']}")
            tlog._log_spaces += 4
            # tuning hyperparameters
            if hasattr(tuner["model"], "set_y_classes"):
                tuner["model"].set_y_classes(
                    mixed_data.get_classes(), mixed_data.get_y_probs()
                )
            tuner["model"].fit(mixed_data.X, mixed_data.y)
            # cross-validate best result
            tlog("Cross-validating best estimator")
            iads_res, pmemo_res = cross_validate(
                get_best_model(tuner),
                self.iads,
                self.pmemo,
                self.splitter,
                [
                    metrics.r2_score,
                    lambda x, y: metrics.mean_squared_error(x, y, squared=False),
                    metrics.mean_absolute_error,
                ],
                label,
            )

            pickle.dump(get_best_model(tuner), open(tuner["name"] + ".pkl", "wb"))

            tlog("\n\n")
            tlog("___________________")
            tlog("Obtained metrics for IADS")
            tlog("   r2, RMSE, MAE")
            for v, err in iads_res:
                tlog(f"{v:.2e} ± {err:.2e}")
            tlog("___________________")
            tlog("Obtained metrics for PMEmo")
            tlog("   r2, RMSE, MAE")
            for v, err in pmemo_res:
                tlog(f"{v:.2e} ± {err:.2e}")
            tlog()
            tlog._log_spaces -= 4


if __name__ == "__main__":
    import fire

    fire.Fire(Main)
