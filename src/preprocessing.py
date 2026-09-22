"""Preprocessing: engineered ratios + sklearn ColumnTransformer.

Pipeline (exactly per spec):
  numeric      -> SimpleImputer(median) + StandardScaler
  categorical  -> SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')
combined via ColumnTransformer into one preprocessing step.

Engineered features (all from application_train.csv columns only):
  CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, CREDIT_ANNUITY_RATIO,
  GOODS_INCOME_RATIO, INCOME_PER_PERSON, AGE_YEARS, EMPLOYMENT_YEARS,
  EMPLOYED_BIRTH_RATIO, EXT_SOURCE_MEAN/STD, CREDIT_TERM_APPROX,
  EMPLOYMENT_BUCKET (categorical bucket of employment length).

Post-encoding width lands ~250-300 (reported exactly at train time).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Sentinel in Home Credit: 365243 means "unemployed / missing"
DAYS_EMPLOYED_SENTINEL = 365243

ENGINEERED_NUMERIC = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_ANNUITY_RATIO",
    "GOODS_INCOME_RATIO",
    "INCOME_PER_PERSON",
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "EMPLOYED_BIRTH_RATIO",
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_STD",
    "CREDIT_TERM_APPROX",
]
ENGINEERED_CATEGORICAL = ["EMPLOYMENT_BUCKET"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add light ratio/aggregation features. Pure function (no fitting).

    Args:
        df: Raw application features (TARGET optional, preserved if present).

    Returns:
        Copy with ~12 extra columns. Infinities -> NaN (handled by imputer).
    """
    out = df.copy()
    eps = 1e-8

    income = out.get("AMT_INCOME_TOTAL", pd.Series(np.nan, index=out.index)).replace(0, np.nan)
    credit = out.get("AMT_CREDIT", pd.Series(np.nan, index=out.index))
    annuity = out.get("AMT_ANNUITY", pd.Series(np.nan, index=out.index))
    goods = out.get("AMT_GOODS_PRICE", pd.Series(np.nan, index=out.index))

    out["CREDIT_INCOME_RATIO"] = credit / (income + eps)
    out["ANNUITY_INCOME_RATIO"] = annuity / (income + eps)
    out["CREDIT_ANNUITY_RATIO"] = credit / (annuity + eps)
    out["GOODS_INCOME_RATIO"] = goods / (income + eps)

    fam = out.get("CNT_FAM_MEMBERS", pd.Series(np.nan, index=out.index)).replace(0, np.nan)
    out["INCOME_PER_PERSON"] = income / (fam + eps)

    birth = out.get("DAYS_BIRTH", pd.Series(np.nan, index=out.index))
    out["AGE_YEARS"] = -birth / 365.0

    emp = out.get("DAYS_EMPLOYED", pd.Series(np.nan, index=out.index)).replace(
        DAYS_EMPLOYED_SENTINEL, np.nan
    )
    out["EMPLOYMENT_YEARS"] = -emp / 365.0
    out["EMPLOYED_BIRTH_RATIO"] = emp / (birth + eps)  # both negative -> positive ratio

    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in out.columns]
    if ext_cols:
        out["EXT_SOURCE_MEAN"] = out[ext_cols].mean(axis=1, skipna=True)
        out["EXT_SOURCE_STD"] = out[ext_cols].std(axis=1, skipna=True)
    else:
        out["EXT_SOURCE_MEAN"] = np.nan
        out["EXT_SOURCE_STD"] = np.nan

    out["CREDIT_TERM_APPROX"] = credit / (annuity + eps)

    # Employment-length buckets -> categorical for OneHotEncoder
    yrs = out["EMPLOYMENT_YEARS"]
    out["EMPLOYMENT_BUCKET"] = pd.cut(
        yrs,
        bins=[-np.inf, 1, 3, 5, 10, np.inf],
        labels=["lt1y", "1-3y", "3-5y", "5-10y", "gt10y"],
    ).astype("object")
    out.loc[yrs.isna(), "EMPLOYMENT_BUCKET"] = np.nan

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def infer_column_types(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    id_col: str = "SK_ID_CURR",
) -> tuple[list[str], list[str]]:
    """Split engineered-frame columns into numeric vs categorical lists.

    Args:
        df: Feature frame AFTER add_engineered_features (TARGET/ID excluded automatically).
        target_col: Label column to exclude.
        id_col: ID column to exclude.

    Returns:
        (numeric_features, categorical_features) as column-name lists.
    """
    feature_cols = [c for c in df.columns if c not in (target_col, id_col)]
    # pandas 2 (object) and pandas 3 (str) compatible categorical detection
    cat = df[feature_cols].select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    num = [c for c in feature_cols if c not in cat]
    return num, cat


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build the single ColumnTransformer required by the spec.

    Numeric: median imputation + StandardScaler.
    Categorical: most-frequent imputation + OneHotEncoder(ignore unknown).
    dense output (sparse_output=False) — required because SMOTE needs dense input
    and TruncatedSVD benchmarking is wall-clocked on the same dense matrix.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def encoded_feature_count(preprocessor: ColumnTransformer) -> int:
    """Return post-encoding width (must be fitted first)."""
    try:
        return len(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback: sum one-hot categories + numeric count
        n = 0
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                n += len(cols)
            elif name == "cat":
                ohe = trans.named_steps["onehot"]
                n += sum(len(c) for c in ohe.categories_)
        return n
