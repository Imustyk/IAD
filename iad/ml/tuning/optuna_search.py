"""Optuna-based hyperparameter search.

Why a custom wrapper rather than ``optuna.integration.OptunaSearchCV``?
    * The CV wrapper requires the *fully constructed* estimator with
      hyperparameters already set; it cannot vary the estimator across trials
      directly. Our wrapper accepts a model name from the registry and
      builds a fresh pipeline per trial (preprocessor + estimator) so each
      trial gets a clean estimator with the trial's suggested params.
    * The result is a ``OptunaSearchResult`` dataclass with the best
      pipeline, the study, the leaderboard and a flag for the underlying
      sampler / pruner. That structure feeds Phase 4 / Phase 5 directly.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from iad.core.exceptions import TrainingError
from iad.core.logging import get_logger
from iad.ml.evaluation.metrics import scoring_for
from iad.ml.preprocessing.pipelines.auto import build_auto_preprocessor
from iad.ml.training.registry import ModelRegistry, Task
from iad.ml.tuning.search_spaces import has_search_space, suggest_params

logger = get_logger("iad.training")


@dataclass(frozen=True)
class OptunaSearchResult:
    """Outcome of :meth:`OptunaSearch.fit`."""

    model_name: str
    task: Task
    best_params: dict[str, Any]
    best_score: float
    best_estimator: Pipeline
    study: optuna.Study
    n_trials: int
    failed_trials: int
    scoring: str
    history: pd.DataFrame = field(default_factory=pd.DataFrame)


class OptunaSearch:
    """Run an Optuna study over a registered model's search space."""

    def __init__(
        self,
        model_name: str,
        task: Task,
        *,
        registry: ModelRegistry | None = None,
        preprocessor: Pipeline | None = None,
        cv: int = 3,
        n_trials: int = 30,
        timeout_seconds: int | None = None,
        scoring: str | None = None,
        random_state: int = 42,
        sampler: optuna.samplers.BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
        study_name: str | None = None,
        direction: Literal["maximize", "minimize"] = "maximize",
    ) -> None:
        self.model_name = model_name
        self.task: Task = task
        self.registry = registry or ModelRegistry.default()
        if (task, model_name) not in self.registry:
            raise KeyError(f"unknown model {model_name!r} for task {task!r}")
        if not has_search_space(model_name):
            raise TrainingError(
                f"no search space registered for {model_name!r}",
                user_message=f"Hyperparameter tuning not supported for model '{model_name}'.",
            )
        self.preprocessor = preprocessor
        self.cv = cv
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        self.scoring = scoring or scoring_for(task)
        self.random_state = random_state
        self.sampler = sampler or optuna.samplers.TPESampler(seed=random_state)
        self.pruner = pruner
        self.study_name = study_name
        self.direction = direction

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: Iterable,
        *,
        callbacks: list | None = None,
    ) -> OptunaSearchResult:
        spec = self.registry.get(self.task, self.model_name)
        preprocessor = self.preprocessor
        if preprocessor is None:
            preprocessor, _ = build_auto_preprocessor(X, target=None, task=self.task)

        # Optuna is verbose by default; quiet it down to WARNING for the
        # platform's own logs (training.log captures the summary line).
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        failed: list[int] = []

        def objective(trial: optuna.trial.Trial) -> float:
            params = suggest_params(self.model_name, trial)
            estimator = spec.factory(**params)
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
            try:
                scores = cross_val_score(pipeline, X, y, cv=self.cv, scoring=self.scoring, n_jobs=1)
                return float(scores.mean())
            except Exception as exc:
                failed.append(trial.number)
                logger.debug("optuna trial %d failed: %s", trial.number, exc)
                raise optuna.TrialPruned() from exc

        study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=self.study_name or f"iad_{self.model_name.lower().replace(' ', '_')}",
        )
        logger.info(
            "optuna search starting",
            extra={
                "ctx_model": self.model_name,
                "ctx_task": self.task,
                "ctx_n_trials": self.n_trials,
                "ctx_timeout_s": self.timeout_seconds,
                "ctx_scoring": self.scoring,
            },
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout_seconds,
            catch=(Exception,),
            callbacks=callbacks,
            n_jobs=1,
            show_progress_bar=False,
        )

        if not study.trials:
            raise TrainingError(
                "Optuna study produced no completed trials",
                user_message="Hyperparameter search failed to complete any trial.",
            )

        best_params = study.best_params
        best_estimator = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", spec.factory(**best_params)),
            ]
        ).fit(X, y)

        history_rows = []
        for trial in study.trials:
            history_rows.append(
                {
                    "trial": trial.number,
                    "value": trial.value,
                    "state": trial.state.name,
                    **trial.params,
                }
            )
        history = pd.DataFrame(history_rows)

        result = OptunaSearchResult(
            model_name=self.model_name,
            task=self.task,
            best_params=best_params,
            best_score=float(study.best_value),
            best_estimator=best_estimator,
            study=study,
            n_trials=len(study.trials),
            failed_trials=len(failed),
            scoring=self.scoring,
            history=history,
        )
        logger.info(
            "optuna search done",
            extra={
                "ctx_model": self.model_name,
                "ctx_best_score": result.best_score,
                "ctx_best_params": result.best_params,
                "ctx_failed_trials": result.failed_trials,
            },
        )
        return result
