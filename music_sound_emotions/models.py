import warnings
from copy import deepcopy

import autosklearn
import numpy as np
from autosklearn.regression import AutoSklearnRegressor
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.experimental import enable_halving_search_cv
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    BaseCrossValidator,
    HalvingGridSearchCV,
    ParameterGrid,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm

from . import settings as S
from .settings import tlog


def _get_pipeline(classifier):
    return Pipeline(
        [("scaler", StandardScaler()), ("pca", PCA()), ("classifier", classifier)]
    )


def get_best_model(tuner: dict):
    """
    Given a tuner (an estimator whose `fit` method tunes the hyper-parameters),
    returns the best model found, with `fit` which tunes the parameters
    (not the hyper-parameters).

    If tuner was not fitted, None is returned.
    """

    m = tuner["model"]
    if isinstance(m, AutoSklearnRegressor):
        return m
    elif isinstance(m, HalvingGridSearchCV):
        params = getattr(m, "best_params_", None)
        if params is None:
            return None
        return clone(m.estimator).set_params(**params)
    else:
        raise TypeError(
            f"Only HalvingGridSearchCV and AutoSklearnRegressor supported, but received {type(m)}."
        )


def get_tuners(splitter: BaseCrossValidator) -> list:
    """
    Given a splitter, returns a list of estimators whose `fit` methods tunes
    hyper-parameters of models
    """
    halving_gridsearch_params = dict(
        factor=4,
        scoring="neg_root_mean_squared_error",
        random_state=1992,
        refit=False,
        min_resources=2 * S.N_SPLITS**2,
        n_jobs=-1,
        verbose=3,
    )
    tuners = [
        {
            "name": "AutoML",
            "model": AutoSklearnRegressor(
                time_left_for_this_task=8 * 3600,
                n_jobs=-1,
                seed=8229,
                memory_limit=10000,
                ensemble_nbest=10,
                metric=autosklearn.metrics.mean_squared_error,
                resampling_strategy="cv",
                resampling_strategy_arguments=dict(shuffle=True, folds=S.N_SPLITS),
            ),
        },
        {
            "name": "Linear",
            "model": CustomHalvingGridSearchCV(
                estimator=_get_pipeline(
                    ElasticNetCV(n_alphas=100, max_iter=10**6, tol=1e-5, cv=5)
                ),
                param_grid=dict(
                    pca__n_components=np.linspace(0.8, 1 - 1e-15, 10),
                    pca__whiten=[True, False],
                    classifier__l1_ratio=np.linspace(0.01, 0.99, 10),
                    classifier__normalize=[True, False],
                ),
                cv=deepcopy(splitter),
                **halving_gridsearch_params,
            ),
        },
        {
            "name": "SVM",
            "model": CustomHalvingGridSearchCV(
                estimator=_get_pipeline(SVR()),
                param_grid=[
                    dict(
                        pca__n_components=np.linspace(0.8, 1 - 1e-15, 5),
                        pca__whiten=[True, False],
                        classifier__kernel=["rbf"],
                        classifier__gamma=["scale", "auto"],
                        classifier__shrinking=[True, False],
                        classifier__C=np.geomspace(0.1, 10.0, 10),
                        classifier__epsilon=np.linspace(0.0, 1.0, 5),
                    ),
                    dict(
                        pca__n_components=np.linspace(0.8, 1 - 1e-15, 5),
                        pca__whiten=[True, False],
                        classifier__kernel=["sigmoid"],
                        classifier__gamma=["scale", "auto"],
                        classifier__shrinking=[True, False],
                        classifier__C=np.geomspace(0.1, 10.0, 10),
                        classifier__coef0=np.linspace(0.0, 100.0, 5),
                        classifier__epsilon=np.linspace(0.0, 1.0, 5),
                    ),
                    dict(
                        pca__n_components=np.linspace(0.8, 1 - 1e-15, 5),
                        pca__whiten=[True, False],
                        classifier__kernel=["poly"],
                        classifier__degree=[2, 3, 4, 5],
                        classifier__gamma=["scale", "auto"],
                        classifier__shrinking=[True, False],
                        classifier__C=np.geomspace(0.1, 10.0, 10),
                        classifier__coef0=np.linspace(0.0, 100.0, 5),
                        classifier__epsilon=np.linspace(0.0, 1.0, 5),
                    ),
                ],
                cv=deepcopy(splitter),
                **halving_gridsearch_params,
            ),
        },
    ]

    return tuners


class CustomHalvingGridSearchCV(HalvingGridSearchCV):
    def set_y_classes(self, y_classes, y_probs):
        self.y_classes = y_classes
        self.y_probs = y_probs

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)
        if isinstance(self.random_state, int):
            self.random_state = np.random.default_rng(self.random_state)

        if not hasattr(self, "y_classes"):
            raise RuntimeError("Please, call `set_classes` first")

        assert self.y_classes.shape == y.shape
        assert self.y_probs.shape == y.shape

        scorer = get_scorer(self.scoring)
        tot = X.shape[0]
        resources = self.min_resources
        best_params = list(ParameterGrid(self.param_grid))
        tlog(f"Total parameter sets: {len(best_params)}")
        i = 0
        with Parallel(n_jobs=self.n_jobs) as parallel:
            while resources < tot:
                i += 1
                scores = []
                idx = self.random_state.choice(
                    np.arange(tot),
                    size=resources,
                    replace=False,
                    p=self.y_probs,
                    shuffle=True,
                )

                def _cv_valid(params):
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        cv_scores = []
                        for train, test in self.cv.split(X[idx], self.y_classes[idx]):
                            estimator = clone(self.estimator)
                            estimator.set_params(**params)
                            estimator.fit(X[train], y[train])
                            score = scorer(estimator, X[test], y[test])
                            cv_scores.append(score)
                        return np.mean(cv_scores)

                bar = tqdm(best_params)
                bar.set_description(tlog._log_spaces * " " +
                                    f"Iteration {i}")
                scores = parallel(delayed(_cv_valid)(params) for params in bar)

                resources = min(tot, 2 * resources)
                # sort best_params by scores
                best_params = [
                    x
                    for _, x in sorted(
                        zip(scores, best_params), key=lambda x: x[0], reverse=True
                    )
                ]
                # keeping only the best parameters
                L = round(len(best_params) / self.factor)
                L = len(best_params) if L <= 1 else L
                best_params = best_params[:L]
                self.best_params_ = best_params[0]
        return self
