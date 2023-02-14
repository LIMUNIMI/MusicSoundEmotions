from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold

from .splits import DataXy, MixedStratifiedKFold


def cross_validate(
    model, data_a: DataXy, data_b: DataXy, splitter: MixedStratifiedKFold, metrics: list
):
    """
    Given a model, cross validates it on the mixed data while testing on
    the separated test folds
    """

    metrics = [get_scorer(m) for m in metrics]

    metrics_a = []
    metrics_b = []
    X, y = splitter.get_full_data()
    for train, test_a, test_b in splitter.custom_split():

        model_ = clone(model)
        model_.fit(X[train], y[train])

        y_a_cap = model_.predict(X[test_a])
        y_a_true = data_a.y[test_a]
        y_b_cap = model_.predict(X[test_b])
        y_b_true = data_b.y[test_b]

        for metric in metrics:
            metrics_a.append(metric(y_a_true, y_a_cap))
            metrics_b.append(metric(y_b_true, y_b_cap))

    return metrics_a, metrics_b


def main(label, p):

    from .data import load_data
    from .models import get_models

    iads, pmemo = load_data()

    splitter = MixedStratifiedKFold(
        iads,
        pmemo,
        p=p,
        base_splitter=StratifiedKFold(n_splits=5, random_state=1983, shuffle=True),
        random_state=1992,
    )
    full_data = splitter.get_full_data()

    for model in get_models(splitter):
        # tuning hyperparameters
        model.fit(full_data.X, full_data.y)
        # cross-validate best result
        iads_res, pmemo_res = cross_validate(
            model,
            iads,
            pmemo,
            splitter,
            ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
        )

        print("\n\n___________________")
        print("Obtained metrics for IADS")
        print("   r2, 1-RMSE, 1-MAE")
        print(f"{i:.2e}" for i in iads_res)
        print("___________________")
        print("Obtained metrics for PMEmo")
        print("   r2, 1-RMSE, 1-MAE")
        print(f"{i:.2e}" for i in pmemo_res)
        print()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
