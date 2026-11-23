"""Обучение и оценка ансамблей."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


# Лучший результат кросс-валидации на train.
SELECTED_MODEL = "random_forest_leaf3"
CANDIDATES = {
    "random_forest_leaf1": {"n_estimators": 250, "min_samples_leaf": 1, "max_features": "sqrt"},
    "random_forest_leaf3": {"n_estimators": 250, "min_samples_leaf": 3, "max_features": "sqrt"},
    "extra_trees_leaf1": {"n_estimators": 300, "min_samples_leaf": 1, "max_features": "sqrt"},
    "extra_trees_leaf2": {"n_estimators": 300, "min_samples_leaf": 2, "max_features": "sqrt"},
    "gradient_boosting_depth1": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 1},
    "gradient_boosting_depth2": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 2},
    "hist_gradient_boosting": {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 2.0, "early_stopping": False},
}


def make_model(name: str, seed: int):
    params = CANDIDATES[name]
    if name.startswith("random_forest"):
        model = RandomForestClassifier(**params, random_state=seed, n_jobs=1)
    elif name.startswith("extra_trees"):
        model = ExtraTreesClassifier(**params, random_state=seed, n_jobs=1)
    elif name.startswith("gradient_boosting"):
        model = GradientBoostingClassifier(**params, random_state=seed)
    else:
        model = HistGradientBoostingClassifier(**params, random_state=seed)
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), model)


def calculate_metrics(target, predictions, probabilities):
    target = np.asarray(target).reshape(-1)
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("ROC-AUC requires both classes 0 and 1 in test_target")
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
    }


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучает выбранный лес и считает метрики на тестовой выборке."""
    train = np.asarray(data.train_features, dtype=float)
    test = np.asarray(data.test_features, dtype=float)
    y = np.asarray(data.train_target)
    target = np.asarray(data.test_target)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    if train.ndim != 2 or test.ndim != 2 or y.ndim != 1 or target.ndim != 1:
        raise ValueError("Features must be matrices and targets must be vectors")
    if (not len(train) or not len(test) or not train.shape[1]
            or train.shape[1] != test.shape[1] or len(train) != len(y) or len(test) != len(target)):
        raise ValueError("Empty data or incompatible dimensions")
    if np.isinf(train).any() or np.isinf(test).any():
        raise ValueError("Infinite feature values are not supported")
    if set(np.unique(y)) != {0, 1} or set(np.unique(target)) != {0, 1}:
        raise ValueError("Train and test targets must contain both binary classes 0 and 1")

    model = make_model(SELECTED_MODEL, data.random_seed)
    model.fit(train, y)
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(test)[:, positive_index]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = calculate_metrics(target, predictions, probabilities)
    estimator = model.steps[-1][1]
    return EnsembleResult(
        model_name=f"{type(estimator).__name__} ({SELECTED_MODEL})",
        probabilities=probabilities, predictions=predictions, **metrics,
    )
