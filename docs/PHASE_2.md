# Phase 2 — Data Engineering Layer

## Goal

Build the production data-engineering layer in `iad/ml/preprocessing/` that
every model in Phase 3 will sit on top of: declarative schema validation,
quality checks, drift detection, sklearn-compatible feature-engineering
transformers and a fluent pipeline builder with an auto-factory.

## What was delivered

```
iad/ml/preprocessing/
├── __init__.py                                # curated public API
├── exceptions.py                              # PreprocessingError, SchemaValidationFailed,
│                                              # DriftDetectionError, TransformerNotFittedError
├── _dtypes.py                                 # is_categorical_like / numeric / datetime helpers
├── schemas/
│   ├── __init__.py
│   ├── base.py                                # validate_with_pandera + SchemaValidationResult
│   ├── samples.py                             # 7 schemas for the bundled sample datasets
│   └── great_expectations_adapter.py          # optional GE bridge (Pandera → ExpectationSuite)
├── quality/
│   ├── __init__.py
│   ├── duplicates.py                          # rows + columns + constant + near-constant
│   ├── nulls.py                               # threshold rules + report
│   ├── outliers.py                            # IQR, Z-score, IsolationForest + cap_outliers
│   └── rules.py                               # ImpossibleValueRule DSL (10 rule types)
├── drift/
│   ├── __init__.py
│   ├── metrics.py                             # KS + PSI + JS + Chi-square
│   └── detector.py                            # DriftDetector + DriftReport
├── profiling/
│   ├── __init__.py
│   └── profiler.py                            # DataProfile + 0-1 quality score + HTML render
├── transformers/
│   ├── __init__.py
│   ├── datetime_features.py                   # DatetimeFeatureExtractor (10 calendar features)
│   ├── rare_category.py                       # RareCategoryGrouper
│   ├── skewness.py                            # SkewnessCorrector (Yeo-Johnson)
│   ├── multicollinearity.py                   # MulticollinearityReducer
│   ├── target_encoder.py                      # SmoothedTargetEncoder (K-fold OOF)
│   └── feature_selector.py                    # AutoFeatureSelector (variance + corr + RF)
└── pipelines/
    ├── __init__.py
    ├── builder.py                             # fluent PreprocessingPipelineBuilder
    └── auto.py                                # build_auto_preprocessor()
```

### Tests

```
tests/unit/preprocessing/
├── test_pandera_schemas.py                    # Pandera + GE adapter
├── test_quality.py                            # duplicates, nulls, outliers, rules
├── test_drift.py                              # KS / PSI / JS / detector flow
├── test_transformers.py                       # all 6 transformers, incl. target-encoder leakage test
├── test_profiler_and_pipelines.py             # profile + builder + auto
└── test_sklearn_contract.py                   # sklearn.clone() round-trip on every transformer

tests/integration/test_phase2_pipeline.py      # end-to-end on Telco + Iris + Phase-1 regression
```

### Test results

* **137 / 137 passing** (114 unit + 13 sklearn-contract + 7 phase-2-specific + 3 integration + Phase-1 carry-over)
* **80% line coverage** across 1,651 statements (>85% on the modules with novel logic — drift, transformers, pipelines)
* Phase-1 regression (legacy training pipeline parity) still green.

## Key design decisions

### ADR-004 — Pandera as the primary schema engine

Lightweight (~1MB), pandas-native, decorator API, integrates with sklearn.
Great Expectations is heavyweight (>50MB transitive deps, designed for
Airflow/Dagster). We ship a thin GE *adapter* (`great_expectations_adapter.py`)
that translates a Pandera schema to a GE `ExpectationSuite` and only
activates when GE is importable.

### ADR-005 — DIY transformers over `category_encoders`

Implementing target encoding, rare-category grouping, etc. ourselves is
~30-50 LOC per transformer, gives full control over edge cases, and keeps
the dependency graph minimal. Specifically: target encoding must do K-fold
OOF in `fit_transform` to avoid leakage; we test that explicitly.

### ADR-006 — sklearn `BaseEstimator` everywhere

Every transformer subclasses `BaseEstimator + TransformerMixin` and exposes
`get_feature_names_out`. They drop into any sklearn `Pipeline`, `GridSearchCV`,
`OptunaSearchCV`, or Phase 3's MLflow logging without adapters.

### ADR-007 — Strict sklearn-clone contract

`__init__` stores parameters verbatim. Any normalisation (`None → []`,
type coercion, mutable defaults) happens in `fit()`. This is enforced by
`tests/unit/preprocessing/test_sklearn_contract.py` which round-trips every
transformer through `sklearn.base.clone()`.

### ADR-008 — Modern dtype detection in `iad.ml.preprocessing._dtypes`

Pandas ≥ 2.1 reports string columns with `dtype=str` rather than `object`.
The legacy check `df[col].dtype == "object"` silently misses them. The
`is_categorical_like` helper handles object / string / category / boolean
uniformly and is reused everywhere.

## Public API

```python
from iad.ml.preprocessing import (
    # Schemas
    SAMPLE_SCHEMAS, validate_with_pandera, get_sample_schema,
    SchemaValidationResult,

    # Quality
    detect_duplicates, null_report, columns_above_null_threshold,
    detect_outliers_iqr, detect_outliers_zscore, detect_outliers_isolation_forest,
    ImpossibleValueRule, check_impossible_values,

    # Drift
    DriftDetector, DriftReport, ColumnDriftResult,

    # Profiling
    DataProfiler, DataProfile,

    # Transformers
    DatetimeFeatureExtractor, RareCategoryGrouper, SkewnessCorrector,
    MulticollinearityReducer, SmoothedTargetEncoder, AutoFeatureSelector,

    # Pipelines
    PreprocessingPipelineBuilder, build_auto_preprocessor,

    # Exceptions
    PreprocessingError, SchemaValidationFailed, DriftDetectionError,
    TransformerNotFittedError,
)
```

## End-to-end demonstration

Verified on the synthetic Telco churn sample dataset:

| Step | Result |
|------|--------|
| Pandera schema validation | ✅ `is_valid=True`, 0 errors |
| Profiler quality score | 0.96 / 1.00 |
| Quality scan | 0 dup rows · 0 cols above null threshold · 40 IsolationForest outliers |
| Impossible-value rules | 0 violations |
| Drift detection on synthetic shift (`monthly_charges + 50`) | `monthly_charges` correctly flagged |
| `build_auto_preprocessor → GradientBoosting` 3-fold CV | **F1_macro 0.715 ± 0.024** |

## How Phase 2 integrates with the rest of the platform

* **Phase 1.** Every module uses `iad.core.get_logger`. Every exception
  inherits from `iad.core.exceptions.IADError` so the global error handler
  in `iad.core.error_handler` already renders Phase-2 failures correctly.
* **Phase 3 (next).** `iad.ml.training` will call
  `build_auto_preprocessor(df, target, task)` instead of building a fresh
  `ColumnTransformer` inline. The legacy `src.predictive` keeps working
  unchanged until the parity tests land.
* **Phase 4.** The Streamlit pages will replace their inline outlier /
  correlation logic with the `iad.ml.preprocessing.quality` and `.drift`
  primitives; the new components will render `DataProfile`, `DriftReport`
  and `OutlierReport` directly.

## Edge cases handled

* **Modern pandas string dtype.** All transformers + profiler + auto-builder
  use `_dtypes.is_categorical_like` so `dtype=str` columns are detected.
* **Target encoder leakage.** `fit_transform` does K-fold OOF; `transform`
  uses the full-data mapping; unseen categories fall back to the global
  target mean. There is an explicit leakage test in `test_transformers.py`.
* **Skewness on negative / zero data.** Yeo-Johnson is used (Box-Cox would
  fail on non-positive values).
* **Sample size for drift.** `DriftDetector` raises a `flag` (not an error)
  when either side is below the configured `min_sample_size`.
* **High-cardinality categoricals in drift.** Skipped with a flag — these
  are usually IDs and PSI / Chi-square are meaningless on them.
* **Impossible-value rule errors.** Unknown columns return a structured
  `ImpossibleValueReport(error=...)` rather than crashing the batch.
* **GE not installed.** Adapter exposes `is_available()` and falls back to
  Pandera silently. `schema_to_expectations` builds the JSON spec with no
  GE dependency at all.
* **`sklearn.clone()` compatibility.** Verified for every transformer.

## Production considerations

* **Drift in production.** Persist a `DriftReport` next to every prediction
  batch. The PSI thresholds (no-drift < 0.1, moderate < 0.25, significant
  ≥ 0.25) follow Siddiqi 2006 and are the credit-risk industry default.
  Override per-feature via `DriftDetector(psi_threshold=...)`.
* **Memory footprint.** All quality reports cap their payloads (e.g. first
  1,000 duplicate-row indices, first 10 rule violations) so they fit in
  Postgres `JSONB` columns when Phase 6 lands.
* **Reproducibility.** Every transformer that uses an RNG accepts a
  `random_state`. `build_auto_preprocessor` derives nothing from
  uninitialised state — the same input deterministically yields the same
  fitted pipeline.

## Rollback

```bash
git checkout pre-phase-2 -- requirements.txt pyproject.toml
rm -rf iad/ml/preprocessing tests/unit/preprocessing tests/integration/test_phase2_pipeline.py docs/PHASE_2.md
pip uninstall -y pandera
```

The platform is back to its post-Phase-1 state. No legacy module was
modified, so the `src/` and `pages/` paths are entirely unaffected.

## What Phase 3 builds on top of this

* `iad.ml.training` — multi-model leaderboard wrapping XGBoost / LightGBM
  / CatBoost / HistGradientBoosting / ExtraTrees / ElasticNet on top of
  `build_auto_preprocessor`.
* `iad.ml.tuning` — Optuna-based hyperparameter search using sklearn's
  `cross_val_score` (already verified to compose with our transformers).
* `iad.ml.explainability` — SHAP global + local explanations, LIME for
  per-row reasoning, waterfall plots.
* `iad.ml.tracking` — MLflow experiment tracking with full lineage from
  the schema groups returned by `build_auto_preprocessor`.
