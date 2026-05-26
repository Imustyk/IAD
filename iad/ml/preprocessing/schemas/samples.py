"""Pandera schemas for the seven curated sample datasets.

Each schema is intentionally permissive (``strict=False``) so a user can keep
extra columns. Range checks come from prior knowledge of each dataset's
domain; tweak them only if the upstream loader changes.
"""
from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

# ---------------------------------------------------------------------------
# Iris
# ---------------------------------------------------------------------------
IRIS_SCHEMA = DataFrameSchema(
    {
        "sepal length (cm)": Column(float, Check.in_range(0, 20), nullable=False),
        "sepal width (cm)": Column(float, Check.in_range(0, 10), nullable=False),
        "petal length (cm)": Column(float, Check.in_range(0, 20), nullable=False),
        "petal width (cm)": Column(float, Check.in_range(0, 10), nullable=False),
        "species": Column(str, Check.isin(["setosa", "versicolor", "virginica"]), nullable=False),
    },
    strict=False,
    coerce=True,
    name="iris",
)


# ---------------------------------------------------------------------------
# Wine
# ---------------------------------------------------------------------------
WINE_SCHEMA = DataFrameSchema(
    {
        "alcohol": Column(float, Check.in_range(5, 25)),
        "wine_class": Column(str, Check.isin(["class_0", "class_1", "class_2"])),
    },
    strict=False,
    coerce=True,
    name="wine",
)


# ---------------------------------------------------------------------------
# Breast cancer
# ---------------------------------------------------------------------------
BREAST_CANCER_SCHEMA = DataFrameSchema(
    {
        "diagnosis": Column(str, Check.isin(["malignant", "benign"]), nullable=False),
    },
    strict=False,
    coerce=True,
    name="breast_cancer",
)


# ---------------------------------------------------------------------------
# Diabetes (regression)
# ---------------------------------------------------------------------------
DIABETES_SCHEMA = DataFrameSchema(
    {
        "disease_progression": Column(float, Check.in_range(0, 500)),
    },
    strict=False,
    coerce=True,
    name="diabetes",
)


# ---------------------------------------------------------------------------
# California housing (regression)
# ---------------------------------------------------------------------------
CALIFORNIA_SCHEMA = DataFrameSchema(
    {
        "MedInc": Column(float, Check.gt(0)),
        "HouseAge": Column(float, Check.in_range(0, 100)),
        "median_house_value": Column(float, Check.gt(0)),
    },
    strict=False,
    coerce=True,
    name="california_housing",
)


# ---------------------------------------------------------------------------
# Titanic-style (synthetic)
# ---------------------------------------------------------------------------
TITANIC_SCHEMA = DataFrameSchema(
    {
        "pclass": Column(int, Check.isin([1, 2, 3])),
        "sex": Column(str, Check.isin(["male", "female"])),
        "age": Column(float, Check.in_range(0, 120), nullable=True),
        "sibsp": Column(int, Check.ge(0)),
        "parch": Column(int, Check.ge(0)),
        "fare": Column(float, Check.ge(0)),
        "embarked": Column(str, Check.isin(["S", "C", "Q"])),
        "survived": Column(int, Check.isin([0, 1])),
    },
    strict=False,
    coerce=True,
    name="titanic_like",
)


# ---------------------------------------------------------------------------
# Telco churn (synthetic)
# ---------------------------------------------------------------------------
TELCO_CHURN_SCHEMA = DataFrameSchema(
    {
        "tenure_months": Column(int, Check.in_range(0, 240)),
        "monthly_charges": Column(float, Check.ge(0)),
        "total_charges": Column(float, Check.ge(0)),
        "contract_type": Column(
            str, Check.isin(["Month-to-month", "One year", "Two year"])
        ),
        "internet_service": Column(str, Check.isin(["DSL", "Fiber optic", "No"])),
        "payment_method": Column(
            str,
            Check.isin(
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
            ),
        ),
        "is_senior_citizen": Column(int, Check.isin([0, 1])),
        "churn": Column(int, Check.isin([0, 1])),
    },
    strict=False,
    coerce=True,
    name="telco_churn",
)


SAMPLE_SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "Iris (classification)": IRIS_SCHEMA,
    "Wine quality (classification)": WINE_SCHEMA,
    "Breast cancer (classification)": BREAST_CANCER_SCHEMA,
    "Diabetes progression (regression)": DIABETES_SCHEMA,
    "California housing (regression)": CALIFORNIA_SCHEMA,
    "Titanic-style (classification)": TITANIC_SCHEMA,
    "Telco churn (classification)": TELCO_CHURN_SCHEMA,
}


def get_sample_schema(name: str) -> pa.DataFrameSchema | None:
    """Return the Pandera schema for a known sample dataset, or None."""
    return SAMPLE_SCHEMAS.get(name)
