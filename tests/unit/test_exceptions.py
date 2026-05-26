"""Exception hierarchy — codes, messages, context, inheritance."""
from __future__ import annotations

from iad.core.exceptions import (
    DataLoadError,
    IADError,
    InferenceError,
    NotTrainedError,
    SchemaError,
    TrainingError,
    UploadError,
    ValidationError,
)


def test_iad_error_default_message_and_code() -> None:
    err = IADError()
    assert err.user_message == "An unexpected error occurred."
    assert err.code == "iad_error"
    assert err.http_status == 500
    assert err.context == {}


def test_iad_error_custom_message_and_context() -> None:
    err = IADError("technical detail", user_message="Sorry!", request_id="abc")
    assert err.message == "technical detail"
    assert err.user_message == "Sorry!"
    assert err.context == {"request_id": "abc"}


def test_validation_error_inherits_iad_error() -> None:
    err = ValidationError("bad column", field="age")
    assert isinstance(err, IADError)
    assert err.code == "validation_error"
    assert err.http_status == 422
    assert err.context == {"field": "age"}


def test_upload_and_schema_errors_are_validation_errors() -> None:
    assert issubclass(UploadError, ValidationError)
    assert issubclass(SchemaError, ValidationError)


def test_training_inference_not_trained_codes() -> None:
    assert TrainingError().code == "training_error"
    assert InferenceError().code == "inference_error"
    assert NotTrainedError().code == "not_trained"
    assert NotTrainedError().http_status == 409


def test_to_dict_serialises_payload() -> None:
    err = DataLoadError("could not parse", user_message="Bad CSV", path="/tmp/x.csv")
    d = err.to_dict()
    assert d["code"] == "data_load_error"
    assert d["context"] == {"path": "/tmp/x.csv"}
