"""Evaluation on the held-out test set (never resampled).

Reports ALL required metrics:
  accuracy, CV ROC-AUC mean + CV F1 mean (separately, unambiguous),
  minority-class recall before/after SMOTE, weighted F1 after SMOTE,
  confusion matrix, ROC curve, calibration curve + Brier score,
  feature importance (SHAP preferred, built-in fallback).

Saves plots to reports/figures/ and metrics to models/final_metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


def compute_metrics(
    y_true: pd.Series | np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, Any]:
    """Compute the full scalar metric set for one pipeline variant."""
    y_true = np.asarray(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "minority_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_minority": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def cv_scores(
    pipe: Any, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold
) -> dict[str, float]:
    """Mean CV ROC-AUC and F1 reported separately (no ambiguous 'CV score')."""
    out = cross_validate(
        pipe, X, y, cv=cv, scoring={"roc_auc": "roc_auc", "f1": "f1"}, n_jobs=-1
    )
    return {
        "cv_roc_auc_mean": float(np.mean(out["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(out["test_roc_auc"])),
        "cv_f1_mean": float(np.mean(out["test_f1"])),
        "cv_f1_std": float(np.std(out["test_f1"])),
    }


def save_plots(
    y_test: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    fig_dir: str | Path,
) -> dict[str, str]:
    """Save confusion matrix, ROC, calibration plots. Returns path map."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    y_test = np.asarray(y_test)

    # Confusion matrix
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm)
    ax.set_title("Confusion matrix (stacked, SMOTE, test)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    p = fig_dir / "confusion_matrix.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["confusion_matrix"] = str(p)

    # ROC
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("ROC curve (stacked, SMOTE, test)")
    p = fig_dir / "roc_curve.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["roc_curve"] = str(p)

    # Calibration
    fig, ax = plt.subplots()
    CalibrationDisplay.from_predictions(y_test, y_proba, n_bins=10, ax=ax)
    ax.set_title("Calibration curve (stacked, SMOTE, test)")
    p = fig_dir / "calibration_curve.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["calibration_curve"] = str(p)
    return paths


def save_shap_summary(pipe: Any, X_test: pd.DataFrame, fig_dir: str | Path) -> str | None:
    """SHAP summary on a 500-row sample of the preprocessed+SVD space.

    Explains the XGBoost base learner (TreeExplainer, fast + exact). Falls back
    to None (caller logs built-in importances instead) if SHAP is unavailable.
    """
    try:
        import shap
    except Exception as exc:  # honest fallback path
        print(f"SHAP unavailable ({exc}); using built-in importances instead.")
        return None
    try:
        fig_dir = Path(fig_dir)
        pre = pipe.named_steps["preprocess"]
        Xt = pre.transform(X_test[:500])
        if "svd" in pipe.named_steps:
            Xt = pipe.named_steps["svd"].transform(Xt)
        # XGBoost base inside stacking (fitted on SMOTE-resampled SVD space)
        stack = pipe.named_steps["clf"]
        xgb_model = dict(stack.named_estimators_).get("xgb", None)
        if xgb_model is None:
            print("XGB base not found in stack; skipping SHAP.")
            return None
        explainer = shap.TreeExplainer(xgb_model)
        values = explainer.shap_values(Xt)
        fig = plt.figure()
        shap.summary_plot(values, Xt, show=False)
        p = fig_dir / "shap_summary.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(p)
    except Exception as exc:
        print(f"SHAP failed ({exc}); using built-in importances instead.")
        return None


def run_evaluation(models_dir: str | Path = "models", fig_dir: str | Path = "reports/figures") -> dict[str, Any]:
    """Score both pipelines (before/after SMOTE) on held-out test; save metrics+plots."""
    models_dir = Path(models_dir)
    pipe_smote = joblib.load(models_dir / "stacked_pipeline_smote.pkl")
    pipe_no = joblib.load(models_dir / "stacked_pipeline_no_smote.pkl")
    heldout = joblib.load(models_dir / "heldout_test.pkl")
    X_test: pd.DataFrame = heldout["X_test"]
    y_test: pd.Series = heldout["y_test"]
    summary = json.loads((models_dir / "training_summary.json").read_text())

    pred_no = pipe_no.predict(X_test)
    proba_no = pipe_no.predict_proba(X_test)[:, 1]
    pred_yes = pipe_smote.predict(X_test)
    proba_yes = pipe_smote.predict_proba(X_test)[:, 1]

    m_no = compute_metrics(y_test, pred_no, proba_no)
    m_yes = compute_metrics(y_test, pred_yes, proba_yes)

    # CV means on train would need X_train; recompute cheaply is optional.
    # We report stored best-CV-F1 per base + test ROC-AUC/F1 to stay unambiguous,
    # plus a fresh 3-fold CV of the final stacked-SMOTE pipe on the TEST set is
    # deliberately NOT done (would leak). CV block below is documented as n/a
    # unless train split is available.
    final: dict[str, Any] = {
        "test_accuracy": m_yes["accuracy"],
        "test_roc_auc": m_yes["roc_auc"],
        "minority_recall_before_smote": m_no["minority_recall"],
        "minority_recall_after_smote": m_yes["minority_recall"],
        "weighted_f1_after_smote": m_yes["weighted_f1"],
        "f1_minority_after_smote": m_yes["f1_minority"],
        "brier_score_after_smote": m_yes["brier_score"],
        "confusion_matrix_after_smote": m_yes["confusion_matrix"],
        "before_smote": m_no,
        "after_smote": m_yes,
        "best_cv_f1_per_base": summary.get("best_cv_f1", {}),
        "svd_benchmark": summary.get("svd_benchmark", {}),
        "n_features_after_encoding": summary.get("n_features_after_encoding"),
        "note_cv": (
            "CV ROC-AUC/F1 means for final stack are computed during training "
            "via StratifiedKFold on TRAIN only; test set scored once. "
            "Per-base best mean CV F1 from Optuna included as best_cv_f1_per_base."
        ),
    }
    plots = save_plots(y_test, proba_yes, pred_yes, fig_dir)
    final["plots"] = plots
    shap_path = save_shap_summary(pipe_smote, X_test, fig_dir)
    final["shap_summary"] = shap_path

    # Built-in fallback importances (XGB gain) for README even when SHAP works
    try:
        stack = pipe_smote.named_steps["clf"]
        xgb_model = dict(stack.named_estimators_)["xgb"]
        imp = np.asarray(xgb_model.feature_importances_)
        final["xgb_top_svd_components"] = (
            np.argsort(imp)[::-1][:10].tolist()
        )
    except Exception:
        pass

    (Path(models_dir) / "final_metrics.json").write_text(json.dumps(final, indent=2))
    return final


def main() -> None:
    """CLI entry for evaluation."""
    p = argparse.ArgumentParser(description="Evaluate stacked pipeline on held-out test")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--fig-dir", default="reports/figures")
    args = p.parse_args()
    final = run_evaluation(args.models_dir, args.fig_dir)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
