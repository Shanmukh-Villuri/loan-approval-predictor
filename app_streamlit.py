"""Streamlit demo for Loan Approval Predictor (clickable portfolio demo).

Run: streamlit run app_streamlit.py
Expects models/stacked_pipeline_smote.pkl (run python -m src.train first).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.preprocessing import add_engineered_features

MODEL_PATH = Path("models/stacked_pipeline_smote.pkl")


@st.cache_resource
def load_model():
    """Cache the fitted pipeline across reruns."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


st.title("Loan Approval Predictor — Home Credit")
st.caption("Stacked XGB + RF + SVC (SMOTE inside folds, SVD-50). Probabilities are calibrated-ish: check Brier score in README.")

model = load_model()
if model is None:
    st.error("Model not found. Train first: python -m src.train --data data/application_train.csv --out models")
    st.stop()

with st.form("applicant"):
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Annual income (AMT_INCOME_TOTAL)", value=150000.0, min_value=0.0)
        credit = st.number_input("Credit amount (AMT_CREDIT)", value=500000.0, min_value=0.0)
        annuity = st.number_input("Annuity (AMT_ANNUITY)", value=25000.0, min_value=0.0)
        goods = st.number_input("Goods price (AMT_GOODS_PRICE)", value=450000.0, min_value=0.0)
    with col2:
        age = st.slider("Age (years)", 18, 75, 35)
        employed = st.slider("Employment length (years)", 0, 40, 5)
        ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5)
        ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.5)
        ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.5)
    submitted = st.form_submit_button("Predict default risk")

if submitted:
    row = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods,
        "DAYS_BIRTH": -age * 365,
        "DAYS_EMPLOYED": -employed * 365,
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "CNT_FAM_MEMBERS": 2.0,
    }
    df = add_engineered_features(pd.DataFrame([row]))
    proba = float(model.predict_proba(df)[0, 1])
    st.metric("Default probability", f"{proba:.3f}")
    st.progress(min(max(proba, 0.0), 1.0))
    st.write("High risk (likely default)" if proba >= 0.5 else "Lower risk (likely repay)")
    st.caption("Demo uses median-imputed defaults for un-entered fields. See README for calibration (Brier) caveats.")
