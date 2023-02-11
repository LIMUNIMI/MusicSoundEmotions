from autosklearn.regression import AutoSklearnRegressor
from sklearn.model_selection import HalvingGridSearchCV

models = [{
        "name": "Linear",
        "model": HalvingGridSearchCV
        },
          {"name": "SVM",
           "model": HalvingGridSearchCV},
          {"name": "AutoML",
           "model": AutoSklearnRegressor}
]


def get_models(splitter):
    # TODO
    return models
