"""Task-type inference for sample datasets and pandas string columns."""
from __future__ import annotations

from src.data_loader import _load_iris, _load_wine
from src.utils import detect_task_type


def test_detect_task_type_pandas_string_dtype() -> None:
    wine = _load_wine()
    assert str(wine["wine_class"].dtype) == "str"
    assert detect_task_type(wine["wine_class"]) == "classification"


def test_detect_task_type_iris_species() -> None:
    iris = _load_iris()
    assert detect_task_type(iris["species"]) == "classification"


def test_detect_task_type_numeric_regression() -> None:
    from src.data_loader import _load_diabetes

    diabetes = _load_diabetes()
    assert detect_task_type(diabetes["disease_progression"]) == "regression"
