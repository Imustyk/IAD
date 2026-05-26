"""Predictive analytics: model training, evaluation, persistence, inference."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


CLASSIFIERS: dict[str, Any] = {
    "Logistic Regression": LogisticRegression(max_iter=2000, n_jobs=None),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
}

REGRESSORS: dict[str, Any] = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "K-Nearest Neighbors": KNeighborsRegressor(),
}


@dataclass
class TrainingReport:
    task_type: str
    target: str
    features: list[str]
    leaderboard: pd.DataFrame
    best_model_name: str
    metrics: dict[str, float]
    cv_metrics: dict[str, float] = field(default_factory=dict)
    confusion_matrix: pd.DataFrame | None = None
    classes: list[str] | None = None
    feature_importance: pd.DataFrame | None = None
    residuals: pd.DataFrame | None = None
    test_predictions: pd.DataFrame | None = None


def build_preprocessor(df: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    numeric_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    try:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    except TypeError:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ]
        )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def _classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None:
        try:
            classes = np.unique(y_true)
            if len(classes) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        except Exception:
            pass
    return {k: round(v, 4) for k, v in metrics.items()}


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "mape_%": float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100),
    }
    return {k: round(v, 4) for k, v in metrics.items()}


def _extract_feature_importance(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame | None:
    try:
        model = pipeline.named_steps["model"]
        preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
        try:
            transformed_names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            transformed_names = feature_names
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            return None
        if len(importances) != len(transformed_names):
            return None
        out = pd.DataFrame({"feature": transformed_names, "importance": importances})
        return out.sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:
        return None


def train_models(
    df: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    task_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    selected_models: list[str] | None = None,
) -> tuple[Pipeline, TrainingReport]:
    """Train candidate models, return the best pipeline and a structured report."""
    data = df[feature_columns + [target]].dropna(subset=[target]).copy()
    X = data[feature_columns]
    y = data[target]

    stratify = y if task_type == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    preprocessor = build_preprocessor(df, feature_columns)
    candidates = CLASSIFIERS if task_type == "classification" else REGRESSORS
    if selected_models:
        candidates = {k: v for k, v in candidates.items() if k in selected_models}
        if not candidates:
            candidates = CLASSIFIERS if task_type == "classification" else REGRESSORS

    leaderboard_rows: list[dict] = []
    best_pipeline: Pipeline | None = None
    best_metrics: dict[str, float] = {}
    best_name: str = ""
    best_score: float = -np.inf

    primary_metric = "f1_macro" if task_type == "classification" else "r2"

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            if task_type == "classification":
                y_proba = (
                    pipeline.predict_proba(X_test)
                    if hasattr(pipeline.named_steps["model"], "predict_proba")
                    else None
                )
                metrics = _classification_metrics(y_test, y_pred, y_proba)
            else:
                metrics = _regression_metrics(y_test, y_pred)
            row = {"model": name, **metrics}
            leaderboard_rows.append(row)
            score = metrics.get(primary_metric, -np.inf)
            if score > best_score:
                best_score = score
                best_pipeline = pipeline
                best_metrics = metrics
                best_name = name
        except Exception as exc:  # pragma: no cover - defensive
            leaderboard_rows.append({"model": name, "error": str(exc)})

    if best_pipeline is None:
        raise RuntimeError("No model could be trained successfully on the given data.")

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        by=primary_metric, ascending=False, na_position="last"
    ).reset_index(drop=True)

    cv_metrics: dict[str, float] = {}
    try:
        cv_scoring = "f1_macro" if task_type == "classification" else "r2"
        scores = cross_val_score(best_pipeline, X, y, cv=cv_folds, scoring=cv_scoring, n_jobs=-1)
        cv_metrics = {
            f"cv_{cv_scoring}_mean": round(float(scores.mean()), 4),
            f"cv_{cv_scoring}_std": round(float(scores.std()), 4),
        }
    except Exception:
        cv_metrics = {}

    confusion = None
    classes = None
    residuals = None
    test_predictions = None
    if task_type == "classification":
        y_pred = best_pipeline.predict(X_test)
        labels = sorted(pd.Series(y).dropna().unique().tolist(), key=lambda x: str(x))
        classes = [str(c) for c in labels]
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        confusion = pd.DataFrame(cm, index=classes, columns=classes)
        test_predictions = X_test.copy()
        test_predictions["actual"] = y_test.values
        test_predictions["predicted"] = y_pred
    else:
        y_pred = best_pipeline.predict(X_test)
        residuals = pd.DataFrame(
            {
                "actual": y_test.values,
                "predicted": y_pred,
                "residual": y_test.values - y_pred,
            }
        )
        test_predictions = X_test.copy()
        test_predictions["actual"] = y_test.values
        test_predictions["predicted"] = y_pred

    importance = _extract_feature_importance(best_pipeline, feature_columns)

    report = TrainingReport(
        task_type=task_type,
        target=target,
        features=feature_columns,
        leaderboard=leaderboard,
        best_model_name=best_name,
        metrics=best_metrics,
        cv_metrics=cv_metrics,
        confusion_matrix=confusion,
        classes=classes,
        feature_importance=importance,
        residuals=residuals,
        test_predictions=test_predictions,
    )
    return best_pipeline, report


def predict(pipeline: Pipeline, new_data: pd.DataFrame, feature_columns: list[str],
            task_type: str) -> pd.DataFrame:
    """Run inference and return a DataFrame with predictions (and probabilities)."""
    aligned = new_data.copy()
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = np.nan
    aligned = aligned[feature_columns]
    preds = pipeline.predict(aligned)
    out = new_data.copy()
    out["prediction"] = preds
    if task_type == "classification" and hasattr(pipeline.named_steps["model"], "predict_proba"):
        proba = pipeline.predict_proba(aligned)
        classes = pipeline.named_steps["model"].classes_
        for i, cls in enumerate(classes):
            out[f"proba_{cls}"] = proba[:, i].round(4)
    return out
