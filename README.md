# Loan Approval Predictor

End-to-end, reproducible prediction of loan default risk (**Home Credit Default Risk**),
optimizing minority-class (high-risk) recall while keeping calibration trustworthy
(Brier score / calibration curve), not just raw accuracy.

> **Honesty note (read first):** Resume-draft targets (89% accuracy, 0.94 CV,
> recall 0.54→0.77, weighted F1 >0.90, 60% SVD speedup) are **targets, not promises**.
> This repo reports **actual** numbers from real runs below. If a target is missed,
> the Results section says so explicitly with reasons. No cherry-picking, no leakage:
> SMOTE lives **only inside training folds** via `imblearn.Pipeline`.

## Dataset (ONE source, exact)

**Choice: Kaggle “Home Credit Default Risk”, `application_train.csv` ONLY.**

- Why: ~122 raw columns, ~8% `TARGET=1` imbalance, expands to ~250–300 after
  one-hot encoding + a handful of engineered ratios. No bureau/previous joins needed.
- Exact source: https://www.kaggle.com/competitions/home-credit-default-risk/data
- File: `application_train.csv` (~307k rows). Place at `data/application_train.csv`
  (gitignored). Free Kaggle account required.
- Download:
  ```powershell
  kaggle competitions download -c home-credit-default-risk -f application_train.csv -p data/
  # then unzip to data/application_train.csv
  ```
  Or download manually from the link above into `data/`.

## Pipeline (matches CV spec exactly)

```
application_train.csv
  -> add_engineered_features (CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, ...)
  -> ColumnTransformer [numeric: median+StandardScaler | categorical: most-frequent+OneHot]
  -> ~250-300 dense features (exact count reported at train time)
  -> TruncatedSVD (randomized, 50 comps) [benchmarked WITH vs WITHOUT, same folds]
  -> SMOTE (inside train folds only, imblearn Pipeline)
  -> XGB / RF / SVC(prob=True) [each tuned with Optuna TPE]
  -> StackingClassifier (LogisticRegression meta-learner)
  -> held-out test: accuracy, CV ROC-AUC/F1, recall before/after, weighted F1,
     confusion matrix, ROC, calibration/Brier, SHAP
```

- **Preprocessing:** `src/preprocessing.py::build_preprocessor` — single
  `ColumnTransformer`, dense output (SMOTE needs dense).
- **SVD:** `TruncatedSVD(n_components=50)` = sklearn’s randomized SVD (“RandomizedSVD”).
  Wall-clock WITH vs WITHOUT on same model + same folds; honest % reported.
- **Imbalance:** `imblearn.Pipeline(preprocess → [svd] → [smote] → clf)`. Test set never resampled.
- **Models:** `XGBClassifier` + `RandomForestClassifier` + `SVC(probability=True)`,
  stacked with `LogisticRegression(max_iter=1000)` — linear combiner is low-variance,
  preserves calibrated probabilities, standard for heterogeneous bases.
- **Tuning:** Optuna TPE (not grid): XGB ~30 trials, RF ~30, SVC ~20, scoring=minority F1
  (pure recall is gameable; recall still reported). SVC tunes/trains on a stratified
  **10% subsample** (`--svc-sample-frac`, documented) — full 245k-row SVC is O(n³) and
  infeasible; XGB/RF use full train.
- **Env:** Python **3.11.9** pinned (`.python-version` + `runtime.txt`), `requirements.txt` pinned.

## Results

> **Status (2026-09-22): PENDING — awaiting the finished training run.**
> Run the pipeline below, then `python -m src.evaluate`; this table is filled in
> with the measured test-set numbers. All figures are reported as measured.

| Metric (held-out test) | Value |
|---|---|
| Accuracy | PENDING |
| ROC-AUC | PENDING |
| CV ROC-AUC mean ± std (train folds) | PENDING |
| CV F1 mean ± std (train folds) | PENDING |
| Minority-class recall, no SMOTE | PENDING |
| Minority-class recall, with SMOTE | PENDING |
| Weighted F1, with SMOTE | PENDING |
| Brier score, with SMOTE | PENDING |
| Features after encoding | 260 |
| SVD train-time change (50 comps, same folds) | PENDING |

Plots (`reports/figures/`): confusion matrix, ROC curve, calibration curve, SHAP summary.

## Notes on the numbers

- The positive (high-risk) class is ~8% of applications, so raw accuracy is a weak
  headline: the operating point is chosen for minority recall at acceptable precision,
  and probabilities are checked via Brier score / calibration curve, not just labels.
- SMOTE is applied inside training folds only; the test set is never resampled, so the
  before/after recall gap is an honest estimate, not leakage.
- SVC trains on a stratified 10% subsample (full 245k-row kernel SVC is infeasible);
  XGB/RF train on the full set. The subsample fraction is recorded in
  `training_summary.json`.
- If compute budget limits the Optuna trial counts, that is stated in
  `training_summary.json` alongside the per-model best mean CV F1 — the tradeoff is
  documented, not hidden.

## Reproduce

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
# place data/application_train.csv (see Dataset above)

# 1) feature-count check + EDA
.\.venv\Scripts\python -m jupyter notebook notebooks/01_eda_prototype.ipynb

# 2) train (tunes XGB/RF/SVC, benchmarks SVD, saves models/*.pkl + training_summary.json)
.\.venv\Scripts\python -m src.train --data data/application_train.csv --out models
# quick smoke test:
.\.venv\Scripts\python -m src.train --quick

# 3) evaluate (saves reports/figures/*.png + models/final_metrics.json)
.\.venv\Scripts\python -m src.evaluate --models-dir models --fig-dir reports/figures

# 4) tests
.\.venv\Scripts\python -m pytest tests -v

# 5) demos
.\.venv\Scripts\python app_cli.py --model models/stacked_pipeline_smote.pkl --input-json sample.json
.\.venv\Scripts\streamlit run app_streamlit.py
```

> **Laptop struggling?** Train on Kaggle instead — see [KAGGLE.md](KAGGLE.md) and
> `notebooks/03_kaggle_full_training.ipynb` (competition data mounts directly, no upload).
> Colab fallback: [COLAB.md](COLAB.md).

## Repo layout

```
src/data_loading.py, preprocessing.py, train.py, evaluate.py
notebooks/01_eda_prototype.ipynb (calls src/)
tests/ (pytest on preprocessing + pipeline structure)
reports/figures/ (confusion_matrix, roc_curve, calibration_curve, shap_summary)
app_cli.py (argparse, test-callable) + app_streamlit.py (clickable demo)
models/ (gitignored artifacts: *.pkl, training_summary.json, final_metrics.json)
data/ (gitignored: application_train.csv)
```

## License

MIT — see LICENSE.
