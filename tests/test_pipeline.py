"""Pipeline structure tests: SVD width, SMOTE placement, stacking shape."""

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.preprocessing import add_engineered_features, build_preprocessor, infer_column_types
from src.train import make_base_estimators, make_imb_pipeline


def _toy(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
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
        # High-cardinality categoricals so post-OHE width exceeds SVD(50) even on toy data
        "ORGANIZATION_TYPE": rng.choice([f"org_{i}" for i in range(40)], n),
        "OCCUPATION_TYPE": rng.choice([f"occ_{i}" for i in range(15)], n),
        "TARGET": rng.choice([0, 1], n, p=[0.85, 0.15]),
    })
    return add_engineered_features(df)


def test_imb_pipeline_order_and_svd_width():
    df = _toy()
    y = df["TARGET"].astype(int)
    X = df.drop(columns=["TARGET"])
    num, cat = infer_column_types(df)
    pre = build_preprocessor(num, cat)
    bases = make_base_estimators(
        {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 10},
        {"C": 1.0, "kernel": "linear", "probability": True},
    )
    stack = StackingClassifier(
        estimators=bases, final_estimator=LogisticRegression(max_iter=500), cv=2
    )
    pipe = make_imb_pipeline(pre, use_svd=True, use_smote=True, estimator=stack)
    assert isinstance(pipe, ImbPipeline)
    names = [n for n, _ in pipe.steps]
    # SMOTE must come after preprocessing/SVD and before classifier (no pre-split leakage)
    assert names == ["preprocess", "svd", "smote", "clf"]
    assert isinstance(pipe.named_steps["svd"], TruncatedSVD)
    assert pipe.named_steps["svd"].n_components == 50
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_no_smote_variant_has_no_smote_step():
    df = _toy()
    num, cat = infer_column_types(df)
    pre = build_preprocessor(num, cat)
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    pipe = make_imb_pipeline(pre, use_svd=True, use_smote=False, estimator=clf)
    assert "smote" not in dict(pipe.steps)
