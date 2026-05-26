"""Machine learning subsystems (Phases 2-3, 11).

Will house:

* ``iad.ml.preprocessing``    — Pandera schemas, sklearn transformers,
                                 feature engineering, profiling, drift checks.
* ``iad.ml.training``         — pipelines, multi-model training, persistence.
* ``iad.ml.evaluation``       — metrics, residuals, calibration, fairness.
* ``iad.ml.explainability``   — SHAP, LIME, global + local explanations.
* ``iad.ml.tuning``           — Optuna hyperparameter optimisation.
* ``iad.ml.tracking``         — MLflow integration (experiments, artifacts).
* ``iad.ml.automl``           — FLAML / PyCaret abstraction layer.
* ``iad.ml.nlp``              — sentiment (VADER), embeddings, LDA topics.
* ``iad.ml.forecasting``      — decomposition, ARIMA, Prophet.
* ``iad.ml.clustering``       — KMeans, DBSCAN, PCA / UMAP.
* ``iad.ml.anomaly``          — Isolation Forest, One-Class SVM.
* ``iad.ml.recommendation``   — collaborative filtering, cosine similarity.
"""
