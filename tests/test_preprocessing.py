"""Smoke tests for preprocessing components."""

import numpy as np
import pandas as pd

from src.preprocessing import (
    add_engineered_features,
    build_preprocessor,
    encoded_feature_count,
    infer_column_types,
)


def _toy_frame(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "SK_ID_CURR": range(n),
        "TARGET": rng.integers(0, 2, n),
        "AMT_INCOME_TOTAL": rng.uniform(5e4, 3e5, n),
        "AMT_CREDIT": rng.uniform(1e5, 1e6, n),
        "AMT_ANNUITY": rng.uniform(5e3, 5e4, n),
        "AMT_GOODS_PRICE": rng.uniform(1e5, 9e5, n),
        "DAYS_BIRTH": -rng.integers(7000, 25000, n),
        "DAYS_EMPLOYED": -rng.integers(100, 8000, n),
        "EXT_SOURCE_1": rng.uniform(0, 1, n),
        "EXT_SOURCE_2": rng.uniform(0, 1, n),
        "EXT_SOURCE_3": rng.uniform(0, 1, n),
        "CNT_FAM_MEMBERS": rng.integers(1, 5, n).astype(float),
        "CODE_GENDER": rng.choice(["M", "F"], n),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
    })


def test_engineered_features_present():
    df = add_engineered_features(_toy_frame())
    for col in ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "EXT_SOURCE_MEAN",
                "AGE_YEARS", "EMPLOYMENT_BUCKET"]:
        assert col in df.columns
    assert (df["CREDIT_INCOME_RATIO"] >= 0).all() or df["CREDIT_INCOME_RATIO"].isna().any()


def test_preprocessor_densifies_and_imputes():
    df = add_engineered_features(_toy_frame())
    num, cat = infer_column_types(df)
    pre = build_preprocessor(num, cat)
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    Xt = pre.fit_transform(X, df["TARGET"])
    assert Xt.shape[0] == len(df)
    assert Xt.shape[1] >= len(num)  # one-hot expands width
    assert not np.isnan(np.asarray(Xt)).any()
    assert encoded_feature_count(pre) == Xt.shape[1]
