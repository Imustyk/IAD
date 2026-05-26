"""Phase 5 ML API route tests."""
from __future__ import annotations

import pytest

from tests.helpers.factories import iris_dataframe


@pytest.mark.integration
def test_train_and_predict_inline(api_client, auth_headers) -> None:
    df = iris_dataframe()
    records = df.sample(40, random_state=42).to_dict(orient="records")
    train = api_client.post(
        "/train",
        headers=auth_headers,
        json={
            "target_column": "species",
            "task_type": "classification",
            "records": records,
            "selected_models": ["Logistic Regression", "Random Forest"],
            "experiment_name": "api_test",
            "persist_to_database": True,
        },
    )
    assert train.status_code == 201, train.text
    body = train.json()
    assert body["best_model_name"]
    assert body["artifact_path"]
    assert len(body["leaderboard"]) >= 1

    artifact = body["artifact_path"]
    predict_rows = df.drop(columns=["species"]).head(3).to_dict(orient="records")
    pred = api_client.post(
        "/predict",
        headers=auth_headers,
        json={"artifact_path": artifact, "records": predict_rows},
    )
    assert pred.status_code == 200, pred.text
    preds = pred.json()["predictions"]
    assert len(preds) == 3
    assert preds[0]["prediction"] is not None


@pytest.mark.integration
def test_upload_train_list_models(api_client, auth_headers) -> None:
    df = iris_dataframe()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    upload = api_client.post(
        "/upload",
        headers=auth_headers,
        files={"file": ("iris.csv", csv_bytes, "text/csv")},
        params={"name": "iris_api"},
    )
    assert upload.status_code == 201, upload.text
    dataset_id = upload.json()["dataset_id"]
    assert dataset_id

    train = api_client.post(
        "/train",
        headers=auth_headers,
        json={
            "target_column": "species",
            "dataset_id": dataset_id,
            "selected_models": ["Logistic Regression"],
            "persist_to_database": True,
        },
    )
    assert train.status_code == 201, train.text
    experiment_id = train.json()["experiment_id"]
    assert experiment_id

    models = api_client.get("/models", headers=auth_headers)
    assert models.status_code == 200
    assert models.json()["total"] >= 1

    experiments = api_client.get("/experiments", headers=auth_headers)
    assert experiments.status_code == 200
    assert any(e["id"] == experiment_id for e in experiments.json()["experiments"])

    metrics = api_client.get(
        "/ml/metrics",
        headers=auth_headers,
        params={"experiment_id": experiment_id},
    )
    assert metrics.status_code == 200
    assert metrics.json()["total"] >= 1


@pytest.mark.integration
def test_train_requires_auth(api_client) -> None:
    response = api_client.post(
        "/train",
        json={"target_column": "y", "records": [{"y": 1, "x": 2}, {"y": 0, "x": 3}]},
    )
    assert response.status_code == 401


@pytest.mark.integration
def test_predict_validation(api_client, auth_headers) -> None:
    response = api_client.post(
        "/predict",
        headers=auth_headers,
        json={"records": [{"a": 1}]},
    )
    assert response.status_code == 422
