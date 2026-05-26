"""Data loading helpers: file uploads, URLs and built-in sample datasets."""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)


@dataclass(frozen=True)
class SampleDataset:
    name: str
    description: str
    suggested_target: str
    loader: Callable[[], pd.DataFrame]


def _load_iris() -> pd.DataFrame:
    raw = load_iris(as_frame=True)
    df = raw.frame.copy()
    df = df.rename(columns={"target": "species"})
    df["species"] = df["species"].map(dict(enumerate(raw.target_names)))
    return df


def _load_wine() -> pd.DataFrame:
    raw = load_wine(as_frame=True)
    df = raw.frame.copy()
    df = df.rename(columns={"target": "wine_class"})
    df["wine_class"] = df["wine_class"].map(dict(enumerate(raw.target_names)))
    return df


def _load_breast_cancer() -> pd.DataFrame:
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df = df.rename(columns={"target": "diagnosis"})
    df["diagnosis"] = df["diagnosis"].map({0: "malignant", 1: "benign"})
    return df


def _load_diabetes() -> pd.DataFrame:
    raw = load_diabetes(as_frame=True)
    df = raw.frame.copy()
    df = df.rename(columns={"target": "disease_progression"})
    return df


def _load_california() -> pd.DataFrame:
    raw = fetch_california_housing(as_frame=True)
    df = raw.frame.copy()
    df = df.rename(columns={"MedHouseVal": "median_house_value"})
    return df


def _make_titanic_like() -> pd.DataFrame:
    """A small synthetic Titanic-style dataset that ships with the app."""
    rng = np.random.default_rng(42)
    n = 600
    pclass = rng.choice([1, 2, 3], size=n, p=[0.2, 0.3, 0.5])
    sex = rng.choice(["male", "female"], size=n, p=[0.6, 0.4])
    age = np.clip(rng.normal(30, 14, size=n), 1, 80).round(1)
    fare = np.where(pclass == 1, rng.normal(80, 30, n),
                    np.where(pclass == 2, rng.normal(25, 10, n),
                             rng.normal(12, 6, n))).clip(min=4).round(2)
    sibsp = rng.poisson(0.5, n)
    parch = rng.poisson(0.4, n)
    embarked = rng.choice(["S", "C", "Q"], size=n, p=[0.7, 0.2, 0.1])

    logits = (
        -1.0
        + (sex == "female") * 2.5
        + (pclass == 1) * 1.0
        + (pclass == 3) * -0.8
        - 0.02 * (age - 30)
        + 0.01 * fare
    )
    proba = 1 / (1 + np.exp(-logits))
    survived = (rng.random(n) < proba).astype(int)

    df = pd.DataFrame(
        {
            "pclass": pclass,
            "sex": sex,
            "age": age,
            "sibsp": sibsp,
            "parch": parch,
            "fare": fare,
            "embarked": embarked,
            "survived": survived,
        }
    )
    mask = rng.random(n) < 0.07
    df.loc[mask, "age"] = np.nan
    return df


def _make_telco_churn_like() -> pd.DataFrame:
    """Synthetic telco churn dataset for classification examples."""
    rng = np.random.default_rng(7)
    n = 800
    tenure = rng.integers(0, 72, n)
    monthly_charges = rng.normal(70, 25, n).clip(15, 200).round(2)
    total_charges = (monthly_charges * np.maximum(tenure, 1)).round(2)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n,
        p=[0.55, 0.25, 0.20],
    )
    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.35, 0.45, 0.20])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n,
    )
    senior = rng.choice([0, 1], size=n, p=[0.85, 0.15])

    logits = (
        -1.5
        + (contract == "Month-to-month") * 1.8
        + (contract == "Two year") * -1.2
        + (internet == "Fiber optic") * 0.7
        + (payment == "Electronic check") * 0.6
        + (senior == 1) * 0.5
        - 0.03 * tenure
        + 0.005 * monthly_charges
    )
    proba = 1 / (1 + np.exp(-logits))
    churn = (rng.random(n) < proba).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract,
            "internet_service": internet,
            "payment_method": payment,
            "is_senior_citizen": senior,
            "churn": churn,
        }
    )


SAMPLE_DATASETS: dict[str, SampleDataset] = {
    "Iris (classification)": SampleDataset(
        name="iris",
        description="Classic Fisher iris flower dataset. Multi-class classification of 3 species.",
        suggested_target="species",
        loader=_load_iris,
    ),
    "Wine quality (classification)": SampleDataset(
        name="wine",
        description="Chemical analysis of wines from three cultivars.",
        suggested_target="wine_class",
        loader=_load_wine,
    ),
    "Breast cancer (classification)": SampleDataset(
        name="breast_cancer",
        description="Wisconsin breast cancer diagnostic dataset (binary).",
        suggested_target="diagnosis",
        loader=_load_breast_cancer,
    ),
    "Diabetes progression (regression)": SampleDataset(
        name="diabetes",
        description="Disease progression one year after baseline.",
        suggested_target="disease_progression",
        loader=_load_diabetes,
    ),
    "California housing (regression)": SampleDataset(
        name="california",
        description="Median house values for California districts.",
        suggested_target="median_house_value",
        loader=_load_california,
    ),
    "Titanic-style (classification)": SampleDataset(
        name="titanic_like",
        description="Synthetic Titanic-style dataset with mixed numeric/categorical features.",
        suggested_target="survived",
        loader=_make_titanic_like,
    ),
    "Telco churn (classification)": SampleDataset(
        name="telco_churn",
        description="Synthetic telecom customer churn dataset for retention analysis.",
        suggested_target="churn",
        loader=_make_telco_churn_like,
    ),
}


def load_sample(name: str) -> pd.DataFrame:
    return SAMPLE_DATASETS[name].loader()


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read a Streamlit UploadedFile into a DataFrame."""
    filename = uploaded_file.name.lower()
    data = uploaded_file.read()
    buffer = io.BytesIO(data)

    if filename.endswith(".csv") or filename.endswith(".txt"):
        try:
            return pd.read_csv(buffer)
        except UnicodeDecodeError:
            buffer.seek(0)
            return pd.read_csv(buffer, encoding="latin-1")
        except pd.errors.ParserError:
            buffer.seek(0)
            return pd.read_csv(buffer, sep=";")
    if filename.endswith(".tsv"):
        return pd.read_csv(buffer, sep="\t")
    if filename.endswith(".json"):
        return pd.read_json(buffer)
    if filename.endswith(".parquet"):
        return pd.read_parquet(buffer)
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    raise ValueError(f"Unsupported file type: {filename}")


def load_from_url(url: str) -> pd.DataFrame:
    """Best-effort URL loader supporting common tabular formats."""
    url_lower = url.lower().split("?")[0]
    if url_lower.endswith(".json"):
        return pd.read_json(url)
    if url_lower.endswith(".parquet"):
        return pd.read_parquet(url)
    if url_lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(url)
    if url_lower.endswith(".tsv"):
        return pd.read_csv(url, sep="\t")
    return pd.read_csv(url)
