import autosklearn
import numpy as np
from autosklearn.regression import AutoSklearnRegressor
from sklearn.decomposition import PCA
from sklearn.experimental import enable_halving_search_cv
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def _get_pipeline(classifier):
    return Pipeline(
        [("scaler", StandardScaler()), ("pca", PCA()), ("classifier", classifier)]
    )


def get_models(splitter):
    halving_gridsearch_params = dict(
        factor=2,
        cv=splitter,
        scoring="neg_root_mean_squared_error",
        random_state=1992,
        refit=False,
        n_jobs=-1,
    )
    models = [
        {
            "name": "Linear",
            "model": HalvingGridSearchCV(
                estimator=_get_pipeline(
                    ElasticNetCV(n_alphas=100, max_iter=10000, tol=1e-10, cv=5)
                ),
                param_grid=dict(
                    pca__n_components=np.linspace(0.8, 1.0, 10),
                    classifier__l1_ratio=np.linspace(0.0, 1.0, 10),
                    classifier__normalize=[True, False],
                ),
                **halving_gridsearch_params
            ),
        },
        {
            "name": "SVM",
            "model": HalvingGridSearchCV(
                estimator=_get_pipeline(SVR()),
                param_grid=[
                    dict(
                        pca__n_components=np.linspace(0.8, 1.0, 10),
                        classifier__kernel=["rbf"],
                        classifier__gamma=["scale", "auto"],
                        classifier__shrinking=[True, False],
                        classifier__C=np.linspace(0.1, 100.0, 20),
                        classifier__epsilon=np.linspace(0.0, 1.0, 10),
                    ),
                    dict(
                        pca__n_components=np.linspace(0.8, 1.0, 10),
                        classifier__kernel=["sigmoid"],
                        classifier__gamma=["scale", "auto"],
                        classifier__shrinking=[True, False],
                        classifier__C=np.linspace(0.1, 100.0, 20),
                        classifier__coef0=np.linspace(0.0, 100.0, 10),
                        classifier__epsilon=np.linspace(0.0, 1.0, 10),
                    ),
                    dict(
                        pca__n_components=np.linspace(0.8, 1.0, 10),
                        classifier__kernel=["poly"],
                        classifier__degree=[2, 3, 4, 5],
                        classifier__gamma=["scale", "auto"],
                        classifier__shrinking=[True, False],
                        classifier__C=np.linspace(0.1, 100.0, 20),
                        classifier__coef0=np.linspace(0.0, 100.0, 10),
                        classifier__epsilon=np.linspace(0.0, 1.0, 10),
                    ),
                ]
                ** halving_gridsearch_params,
            ),
        },
        {
            "name": "AutoML",
            "model": AutoSklearnRegressor(
                time_left_for_this_task=8 * 3600,
                n_jobs=-1,
                seed=8229,
                memory_limit=10000,
                ensemble_nbest=10,
                metric=autosklearn.metrics.mean_squared_error,
                resampling_strategy=splitter,
            ),
        },
    ]

    return models
