"""CLI inference for Loan Approval Predictor.

Reproducible / scriptable prediction entrypoint (also exercised by tests).

Examples:
  python app_cli.py --model models/stacked_pipeline_smote.pkl --input-json sample.json
  python app_cli.py --model models/stacked_pipeline_smote.pkl --input-csv new_applicants.csv --out-csv preds.csv
  # sample.json format: {"AMT_INCOME_TOTAL": 150000, "AMT_CREDIT": 500000, ...}
  # Missing columns are tolerated (pipeline imputes); unknown categories ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import add_engineered_features


def load_model(model_path: str | Path):
    """Load a fitted imblearn pipeline."""
    return joblib.load(model_path)


def predict_df(model, df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features then return proba + label per row."""
    feats = add_engineered_features(df)
    proba = model.predict_proba(feats)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = df.copy()
    out["default_proba"] = proba
    out["default_pred"] = pred
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry."""
    p = argparse.ArgumentParser(description="Predict loan default risk")
    p.add_argument("--model", default="models/stacked_pipeline_smote.pkl")
    p.add_argument("--input-json", default=None, help="Path to single-row JSON object")
    p.add_argument("--input-csv", default=None, help="Path to CSV with raw feature columns")
    p.add_argument("--out-csv", default=None, help="Where to write CSV predictions")
    args = p.parse_args(argv)

    model = load_model(args.model)
    if args.input_json:
        row = json.loads(Path(args.input_json).read_text())
        df = pd.DataFrame([row])
    elif args.input_csv:
        df = pd.read_csv(args.input_csv)
    else:
        p.error("Provide --input-json or --input-csv")
        return 2

    result = predict_df(model, df)
    if args.out_csv:
        result.to_csv(args.out_csv, index=False)
        print(f"Wrote {len(result)} predictions to {args.out_csv}")
    else:
        print(result[["default_proba", "default_pred"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
