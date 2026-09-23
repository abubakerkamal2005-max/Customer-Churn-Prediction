"""
src/preprocessing.py
=====================
Data loading, cleaning, feature engineering, and preprocessing-pipeline
construction for the Customer Churn Prediction system.

Design goals
------------
- Robust to small variations in column naming/casing.
- No data leakage: the ColumnTransformer built here is only ever *fit*
  on training data (see train.py); at inference time it is only *applied*.
- All cleaning/engineering decisions are documented inline and returned
  as a human-readable log so they show up in the training summary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_CANDIDATES = ["churn"]
ID_CANDIDATES = ["customerid", "customer_id", "id"]

# Columns that should always be treated as engineered/derived rather than
# fed straight from the raw file, even though they look numeric/categorical.
ENGINEERED_NUMERIC = ["avg_monthly_charge", "total_services"]
ENGINEERED_CATEGORICAL = ["tenure_group"]


@dataclass
class DatasetProfile:
    """Result of automatic dataset inspection."""

    target_column: str
    id_columns: list[str]
    numerical_columns: list[str]
    categorical_columns: list[str]
    n_rows: int
    n_cols: int
    n_duplicates: int
    missing_summary: dict = field(default_factory=dict)
    target_distribution: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def load_data(path: str) -> pd.DataFrame:
    """Load the raw churn dataset from CSV."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _find_target_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.strip().lower() in TARGET_CANDIDATES:
            return col
    raise ValueError(
        "Could not automatically identify the target ('Churn') column. "
        f"Available columns: {list(df.columns)}"
    )


def _find_id_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.strip().lower() in ID_CANDIDATES]


def inspect_dataset(df: pd.DataFrame) -> DatasetProfile:
    """
    Automatically profile the raw dataset: identify the target column,
    identifier columns, numerical vs. categorical feature columns, and
    produce a concise data-quality report.
    """
    notes: list[str] = []

    target_col = _find_target_column(df)
    id_cols = _find_id_columns(df)

    # TotalCharges (and similar) may be read as object/string because of
    # blank entries even though it is conceptually numeric. Detect these.
    candidate_cols = [c for c in df.columns if c not in id_cols + [target_col]]

    numerical_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in candidate_cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numerical_cols.append(col)
            continue

        # Try coercing object columns that are "secretly" numeric
        # (e.g. TotalCharges stored as strings with blank entries).
        coerced = pd.to_numeric(series.astype(str).str.strip(), errors="coerce")
        non_null_original = series.notna().sum()
        coercible_ratio = coerced.notna().sum() / max(non_null_original, 1)

        if coercible_ratio > 0.95:
            numerical_cols.append(col)
            notes.append(
                f"'{col}' looks numeric but was stored as text "
                f"(coercible ratio={coercible_ratio:.2%}); treated as numerical."
            )
        else:
            categorical_cols.append(col)

    missing_summary = df.isna().sum()
    missing_summary = {k: int(v) for k, v in missing_summary.items() if v > 0}

    n_duplicates = int(df.duplicated().sum())

    target_distribution = df[target_col].value_counts(dropna=False).to_dict()
    target_distribution = {str(k): int(v) for k, v in target_distribution.items()}

    if id_cols:
        notes.append(
            f"Identifier column(s) {id_cols} detected and excluded from modeling "
            "to avoid data leakage (a unique ID has no real predictive signal "
            "and would only let the model memorize training rows)."
        )

    return DatasetProfile(
        target_column=target_col,
        id_columns=id_cols,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        n_rows=df.shape[0],
        n_cols=df.shape[1],
        n_duplicates=n_duplicates,
        missing_summary=missing_summary,
        target_distribution=target_distribution,
        notes=notes,
    )


def clean_data(df: pd.DataFrame, profile: DatasetProfile) -> tuple[pd.DataFrame, list[str]]:
    """
    Clean the raw dataframe according to the inspected profile.

    Returns the cleaned dataframe and a log of cleaning actions taken.
    """
    log: list[str] = []
    df = df.copy()

    # 1. Drop exact duplicate rows.
    if profile.n_duplicates > 0:
        before = len(df)
        df = df.drop_duplicates()
        log.append(f"Dropped {before - len(df)} exact duplicate row(s).")

    # 2. Normalize whitespace-only strings to NaN across object columns,
    #    then coerce columns identified as "secretly numeric" to numeric.
    for col in profile.numerical_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            before_blank = (df[col].astype(str).str.strip() == "").sum()
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
            if before_blank > 0:
                log.append(
                    f"'{col}' contained {before_blank} blank/invalid value(s); "
                    "converted to NaN. These are handled downstream by the "
                    "numerical imputer (median strategy) inside the preprocessing "
                    "pipeline, not silently dropped."
                )

    # 3. Strip whitespace in categorical columns and standardize obvious
    #    "No <service>"-style variants used by the Telco dataset so that
    #    e.g. 'No internet service' collapses sensibly at encoding time.
    for col in profile.categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 4. Target normalization is handled separately (encode_target) to
    #    keep it an explicit, reproducible step rather than a one-off.

    return df, log


def encode_target(df: pd.DataFrame, target_col: str) -> tuple[pd.Series, dict]:
    """
    Convert the target column to a binary 0/1 representation.
    Returns the encoded series and the mapping used (for documentation).
    """
    raw = df[target_col].astype(str).str.strip()
    uniques = sorted(raw.unique())

    if set(uniques) <= {"0", "1"}:
        mapping = {"0": 0, "1": 1}
    else:
        # Standard Yes/No style target.
        positive_tokens = {"yes", "y", "true", "1", "churn"}
        mapping = {}
        for u in uniques:
            mapping[u] = 1 if u.lower() in positive_tokens else 0

    encoded = raw.map(mapping).astype(int)
    return encoded, mapping


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add a small number of justified derived features.

    - avg_monthly_charge: TotalCharges / tenure (falls back to
      MonthlyCharges for customers with tenure == 0, i.e. brand-new
      customers who have no TotalCharges history yet). Captures long-run
      spend rate distinct from the current MonthlyCharges snapshot.
    - tenure_group: coarse tenure buckets. Churn risk is known to be
      highly non-linear in tenure (new customers churn far more than
      long-tenured ones); bucketing gives tree/linear models an easy
      categorical signal alongside the raw numeric value.
    - total_services: count of subscribed add-on services. A simple,
      interpretable proxy for how "embedded" a customer is in the
      product ecosystem.
    """
    log: list[str] = []
    df = df.copy()

    lower_cols = {c.lower(): c for c in df.columns}

    tenure_col = lower_cols.get("tenure")
    monthly_col = lower_cols.get("monthlycharges")
    total_col = lower_cols.get("totalcharges")

    if tenure_col and monthly_col and total_col:
        safe_tenure = df[tenure_col].replace(0, np.nan)
        avg_charge = df[total_col] / safe_tenure
        avg_charge = avg_charge.fillna(df[monthly_col])
        df["avg_monthly_charge"] = avg_charge
        log.append(
            "Engineered 'avg_monthly_charge' = TotalCharges / tenure "
            "(falls back to MonthlyCharges when tenure == 0) to capture "
            "long-run spend rate."
        )

        bins = [-1, 6, 12, 24, 48, np.inf]
        labels = ["0-6mo", "7-12mo", "13-24mo", "25-48mo", "48mo+"]
        df["tenure_group"] = pd.cut(df[tenure_col], bins=bins, labels=labels)
        df["tenure_group"] = df["tenure_group"].astype(str)
        log.append(
            "Engineered 'tenure_group' as coarse tenure buckets to expose "
            "the known non-linear relationship between tenure and churn."
        )

    service_flag_names = [
        "phoneservice",
        "multiplelines",
        "onlinesecurity",
        "onlinebackup",
        "deviceprotection",
        "techsupport",
        "streamingtv",
        "streamingmovies",
    ]
    service_cols = [lower_cols[n] for n in service_flag_names if n in lower_cols]

    if service_cols:

        def _count_services(row):
            count = 0
            for c in service_cols:
                val = str(row[c]).strip().lower()
                if val not in ("no", "no internet service", "no phone service", "nan", ""):
                    count += 1
            return count

        df["total_services"] = df[service_cols].apply(_count_services, axis=1)
        log.append(
            f"Engineered 'total_services' = count of subscribed add-on services "
            f"across {len(service_cols)} service columns."
        )

    return df, log


def build_feature_lists(
    profile: DatasetProfile, engineered_added: bool
) -> tuple[list[str], list[str]]:
    """Build the final numerical/categorical feature lists used for modeling."""
    numerical = list(profile.numerical_columns)
    categorical = list(profile.categorical_columns)

    if engineered_added:
        for f in ENGINEERED_NUMERIC:
            if f not in numerical:
                numerical.append(f)
        for f in ENGINEERED_CATEGORICAL:
            if f not in categorical:
                categorical.append(f)

    return numerical, categorical


def build_preprocessor(numerical_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """
    Build the scikit-learn ColumnTransformer used for both training and
    inference. This object is fit ONLY on training data (see train.py)
    and is saved as part of the final pipeline artifact, so the exact
    same transformation is applied at prediction time.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor
