from sklearn import metrics
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from .splits import DataXy, MixedStratifiedKFold

import numpy as np
import scipy


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
    df = n-1
    t_val = scipy.stats.t.ppf((1 + conf_level) / 2, df)
    moe = t_val * s / np.sqrt(n)
    return moe


def set_label(label, *datasets):
    for dataset in datasets:
        dataset.set_label(label)


def main(label, p):

    from . import settings as S
    from .data import load_data
    from .models import get_best_model, get_tuners

    print("Loading data")
    iads, pmemo = load_data()

    splitter = MixedStratifiedKFold(
        iads,
        pmemo,
        p=p,
        base_splitter=StratifiedKFold(
            n_splits=S.N_SPLITS, random_state=1983, shuffle=True
        ),
        random_state=1992,
    )
    full_data = splitter.get_full_data()

    set_label(label, iads, pmemo, full_data)

    for tuner in get_tuners(splitter):
        print(f"Tuning {tuner['name']}")
        # tuning hyperparameters
        tuner["model"].fit(full_data.X, full_data.y)
        # cross-validate best result
        print("Cross-validating best estimator")
        iads_res, pmemo_res = cross_validate(
            get_best_model(tuner),
            iads,
            pmemo,
            splitter,
            [
                metrics.r2_score,
                lambda x, y: metrics.mean_squared_error(x, y, squared=False),
                metrics.mean_absolute_error,
            ],
            label,
        )

        print("\n\n___________________")
        print("Obtained metrics for IADS")
        print("   r2, RMSE, MAE")
        for v, err in iads_res:
            print(f"{v:.2e} ± {err:.2e}")
        print("___________________")
        print("Obtained metrics for PMEmo")
        print("   r2, RMSE, MAE")
        for v, err in pmemo_res:
            print(f"{v:.2e} ± {err:.2e}")
        print()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
