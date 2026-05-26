"""Optuna search spaces per registered model.

Each function takes an Optuna ``trial`` and returns a hyperparameter dict
appropriate for that model's constructor. Adding tuning support for a new
model = adding one entry to ``_SEARCH_SPACES``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import optuna

SearchSpace = Callable[[optuna.trial.Trial], dict[str, Any]]


def _logistic_regression(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l2"]),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
    }


def _random_forest(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def _gradient_boosting(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=25),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }


def _hist_gradient_boosting(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
    }


def _extra_trees(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def _decision_tree(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
    }


def _knn(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_categorical("p", [1, 2]),
    }


def _ridge(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {"alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}


def _elasticnet(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "max_iter": 5000,
    }


def _xgboost(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _lightgbm(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _catboost(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 100, 800, step=50),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }


_SEARCH_SPACES: dict[str, SearchSpace] = {
    "Logistic Regression": _logistic_regression,
    "Random Forest": _random_forest,
    "Gradient Boosting": _gradient_boosting,
    "Hist Gradient Boosting": _hist_gradient_boosting,
    "Extra Trees": _extra_trees,
    "Decision Tree": _decision_tree,
    "K-Nearest Neighbors": _knn,
    "Ridge Regression": _ridge,
    "ElasticNet": _elasticnet,
    "XGBoost": _xgboost,
    "LightGBM": _lightgbm,
    "CatBoost": _catboost,
}


def has_search_space(model_name: str) -> bool:
    return model_name in _SEARCH_SPACES


def suggest_params(model_name: str, trial: optuna.trial.Trial) -> dict[str, Any]:
    if model_name not in _SEARCH_SPACES:
        raise KeyError(f"no search space registered for model {model_name!r}")
    return _SEARCH_SPACES[model_name](trial)
