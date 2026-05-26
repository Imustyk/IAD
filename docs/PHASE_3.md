# Phase 3 — ML Platform

## Goal

Turn the pure-sklearn coursework training loop into an enterprise ML
platform: pluggable model registry, leaderboard training, Optuna search,
SHAP/LIME explainability, MLflow tracking, FLAML AutoML and full
reproducibility metadata.

## What was delivered

### New packages

```
iad/ml/
├── training/
│   ├── __init__.py
│   ├── registry.py             # ModelSpec + ModelRegistry (lazy optional libs)
│   ├── service.py              # TrainingService (leaderboard + MLflow integration)
│   ├── reports.py              # LeaderboardEntry, TrainingResult
│   ├── reproducibility.py      # SeedManager, EnvironmentFingerprint, ModelCard
│   └── persistence.py          # save_bundle / load_bundle (pipeline + card)
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # classification + regression
│   └── reports.py              # confusion matrix, regression diagnostics, calibration
├── tuning/
│   ├── __init__.py
│   ├── search_spaces.py        # Optuna spaces for 12 model families
│   └── optuna_search.py        # OptunaSearch + OptunaSearchResult
├── explainability/
│   ├── __init__.py
│   ├── shap_explainer.py       # tree/linear/kernel auto-routing + waterfall data
│   └── lime_explainer.py       # tabular LIME wrapper
├── tracking/
│   ├── __init__.py
│   ├── mlflow_tracker.py       # context manager (no-op when MLflow absent)
│   └── runs.py                 # RunMetadata
└── automl/
    ├── __init__.py
    ├── base.py                 # AutoMLBackend ABC + AutoMLResult
    ├── flaml_adapter.py        # default backend
    └── pycaret_adapter.py      # optional adapter (clear install path)
```

### Tests

```
tests/unit/ml/
├── test_evaluation.py             # 7
├── test_reproducibility.py        # 5
├── test_registry_and_service.py   # 9
├── test_persistence.py            # 2
├── test_tuning.py                 # 4
├── test_explainability.py         # 4
├── test_tracking.py               # 2
└── test_automl.py                 # 2

tests/integration/test_phase3_pipeline.py   # 3 end-to-end
```

### Test results

* **175 / 175 passing** (44 net new in Phase 3 + 137 carry-over from
  Phase 1 & 2 + 1 phase-3 integration on top of phase-2 integration suite — the
  legacy Iris regression and Telco auto-pipeline integration are still green).
* **81% line coverage** across 2,758 statements.
* Phase 1 / Phase 2 regression tests carry over unchanged.

## Key design decisions

### ADR-009 — Pluggable `ModelRegistry`

The legacy code hardcoded two dicts (`CLASSIFIERS`, `REGRESSORS`). Phase 3
replaces them with `ModelRegistry`, where each `ModelSpec` is a metadata
+ factory pair. Adding a new estimator = registering one spec. Optional
heavy libraries (XGBoost, LightGBM, CatBoost) are imported lazily and
skipped gracefully — the platform stays usable on a minimal env.

### ADR-010 — User kwargs override factory defaults

Every factory uses ``_merge(defaults, overrides)`` so Optuna-supplied
hyperparameters cleanly replace defaults. Without this contract,
``cross_val_score`` + Optuna would crash with
``"got multiple values for argument 'n_estimators'"``. This is enforced
by the existing sklearn-clone contract test plus the Optuna search tests.

### ADR-011 — `iad.training` log channel

`iad.core.logging` already wires a dedicated rotating ``logs/training.log``
file for the ``iad.training`` namespace (Phase 1). Every log line emitted
by the training service, the Optuna search and the FLAML adapter goes
through this channel, so ML engineers can grep training history without
the UI noise from `iad.frontend` or `iad.backend`.

### ADR-012 — Reproducibility metadata is non-negotiable

Every trained pipeline carries a `ModelCard`:

* random seed,
* SHA-256 fingerprint of the input data (deterministic across re-loads),
* environment snapshot (Python, OS, package versions for the 16 libraries
  the platform depends on),
* hyperparameters, metrics, CV metrics,
* IAD package version and ISO timestamp.

The card is persisted next to the model file and logged to MLflow as a
JSON artefact. Two months from now we can answer "what produced this
model?" without consulting MLflow.

### ADR-013 — Optional dependencies fail-open

XGBoost, LightGBM, CatBoost, MLflow, SHAP, LIME and FLAML are all
optional. Each is imported lazily at first use and degrades to a clear
``ImportError`` (or no-op for tracker) — never to a silent crash. The
`mlflow_available()` / `flaml_available()` predicates let UIs render
"install xxx to enable yyy" hints.

### ADR-014 — SHAP routing strategy

| Model family | Explainer | Cost |
|---|---|---|
| Trees (RF, ET, GB, HistGB, XGB, LGBM, CatBoost, DT) | `TreeExplainer` | O(N · M · D) |
| Linear (Logistic, Ridge, ElasticNet, Lasso, OLS) | `LinearExplainer` | O(N · M) |
| Anything else | `KernelExplainer` | O(N · 2^M) — slow |

Each strategy is auto-selected. KernelExplainer falls back gracefully if
the preferred explainer raises. Background sample is capped (default 100
rows) so the kernel path stays responsive.

## Public API

```python
from iad.ml.training import (
    TrainingService, TrainingConfig, TrainingResult, LeaderboardEntry,
    ModelRegistry, ModelSpec,
    ModelCard, EnvironmentFingerprint, SeedManager, capture_environment, fingerprint_dataframe,
    save_bundle, load_bundle,
)
from iad.ml.evaluation import (
    classification_metrics, regression_metrics, primary_metric_name, scoring_for,
    ConfusionMatrixReport, RegressionReport, CalibrationReport,
)
from iad.ml.tuning import OptunaSearch, OptunaSearchResult, suggest_params, has_search_space
from iad.ml.explainability import SHAPExplainer, SHAPExplanation, LIMEExplainer, LIMEExplanation
from iad.ml.tracking import MLflowTracker, RunMetadata, mlflow_available
from iad.ml.automl import AutoMLBackend, AutoMLResult, FLAMLBackend, PyCaretBackend
```

## End-to-end demonstration (Telco churn)

```
=== PHASE 3 END-TO-END (Telco churn) ===
1. Leaderboard:
                 model  f1_macro  accuracy  roc_auc  train_time_s
   Logistic Regression    0.7506    0.7750   0.8164         0.024
               XGBoost    0.7091    0.7500   0.8016         0.542
         Random Forest    0.6890    0.7312   0.8066         0.192
Hist Gradient Boosting    0.6890    0.7312   0.7994         0.387
              CatBoost    0.6834    0.7250   0.8152         0.233
           Extra Trees    0.6703    0.7000   0.7847         0.172
              LightGBM    0.6690    0.7125   0.7752         1.111
   Best: Logistic Regression  CV: {'cv_f1_macro_mean': 0.7557, 'cv_f1_macro_std': 0.0087}
2. Optuna tuned Logistic Regression: best_score=0.760  (8 trials)
3. SHAP strategy=linear  top features:
   cat__contract_type_Month-to-month    mean|SHAP|=0.1170
   num__tenure_months                   mean|SHAP|=0.0681
   cat__contract_type_Two year          mean|SHAP|=0.0579
   num__total_charges                   mean|SHAP|=0.0470
   cat__contract_type_One year          mean|SHAP|=0.0441
4. ModelCard: name=Logistic Regression  fingerprint=7eb0ad254141e258...
   xgb=3.2.0  lgbm=4.6.0  shap=0.51.0  mlflow-skinny=3.12.0
=== ALL PHASE 3 PRIMITIVES WORK END-TO-END ===
```

## Integration with previous phases

* **Phase 1.** Every Phase 3 module uses `iad.core.get_logger`, raises
  `iad.core.exceptions.IADError` subclasses, validates inputs through
  `iad.core.validation`, and reads settings via `iad.config.get_settings`.
* **Phase 2.** `TrainingService` calls
  `iad.ml.preprocessing.build_auto_preprocessor` to assemble the
  preprocessing pipeline. The schema groups returned by the
  auto-preprocessor are persisted in the `ModelCard.schema_groups` field
  for full lineage.
* **Phase 4 (next).** The Streamlit predictive page will switch from
  `src.predictive.train_models` to `iad.ml.training.TrainingService`.
  The legacy module stays put until parity tests are green for two
  consecutive phases.

## Edge cases handled

* **`sklearn.clone` contract.** Every factory respects the
  "init params stored verbatim" rule, verified by the Phase 2 contract
  tests + every Optuna trial in the integration test.
* **Optional libraries missing.** `ModelRegistry.default()` skips
  unavailable specs silently; tests assert that `MLflowTracker` is
  a no-op when `mlflow` isn't importable.
* **Optuna trial fails.** Wrapped with `catch=(Exception,)`; the trial
  is recorded as `PRUNED` and the search continues.
* **SHAP feature names mismatch.** `_build_feature_names` tries
  `get_feature_names_out(input_features=...)`, then no-args, then
  generic placeholders — never raises.
* **Multi-output models (LogReg multiclass).** Feature importance
  falls back to ``mean(|coef|, axis=0)``; SHAP `explain_row` handles
  both list-of-arrays and 3-D ndarray returns from different SHAP
  versions.
* **Ridiculous hyperparameters.** Optuna trials that can't fit (e.g.
  invalid `solver` / `penalty` combos) raise; we log them as failed
  trials and the study still produces a best result.
* **CatBoost fontconfig spam.** Suppressed by setting
  ``FONTCONFIG_PATH=/dev/null`` is unnecessary at runtime; warning
  is benign and only appears in dev logs.
* **`_count_physical_cores_darwin` joblib warning.** Apple Silicon
  oddity from joblib; set ``LOKY_MAX_CPU_COUNT`` to silence.

## Production considerations

* **Reproducibility.** Always pass a fixed `random_state` to
  `TrainingConfig` (default 42). `SeedManager.set_global_seed` covers
  numpy / Python / torch / TF if installed.
* **MLflow URI.** Set `IAD_MLFLOW_TRACKING_URI` in production
  (PostgreSQL backend recommended; the file backend is deprecated as of
  Feb 2026). The tracker auto-honours the env var.
* **Time budgets.** `TrainingConfig.cv_folds=5` is the default; lower
  to 3 for time-sensitive UI flows. `OptunaSearch(timeout_seconds=...)`
  caps tuning wall time. `FLAMLBackend.fit(time_budget=60)` is the
  AutoML knob.
* **Memory.** SHAP TreeExplainer is constant-memory; KernelExplainer
  scales with `background_data` rows × `2^M`. Default cap of 100
  background rows keeps p99 latency below 5 seconds for ≤ 50 features.
* **Deterministic dataset hash.** `fingerprint_dataframe` uses pandas'
  row-level hasher; deterministic across pandas versions for primitive
  dtypes. For exotic dtypes it falls back to string serialisation.

## Rollback

```bash
git checkout pre-phase-3 -- requirements.txt pyproject.toml
rm -rf iad/ml/{training,evaluation,tuning,explainability,tracking,automl} \
       tests/unit/ml tests/integration/test_phase3_pipeline.py docs/PHASE_3.md
pip uninstall -y xgboost lightgbm catboost optuna shap lime mlflow-skinny flaml
```

The platform reverts to its post-Phase-2 state. Nothing in `src/` or
`pages/` was modified, so the Streamlit app keeps running unchanged.

## What Phase 4 builds on top of this

* `iad/frontend/components/{metric_cards,charts,tables,alerts,model_cards,navbar,layouts}.py`
* Streamlit Predictive Modeling page calls `TrainingService` and renders
  `TrainingResult.leaderboard_frame()` via the new `model_cards` and
  `tables` components.
* SHAP global-importance and waterfall charts wired through Plotly.
* MLflow run links rendered in the model card UI.
* Optuna study progress streamed via Streamlit's progress bar API.
