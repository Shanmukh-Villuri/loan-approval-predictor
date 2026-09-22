# Training on Colab (instead of the laptop)

The full pipeline (Optuna 30/30/20 trials, 3-fold CV, stacking on 307k rows) takes
**hours** and pegs all cores — Colab's high-RAM CPU runtime is a better fit than a
laptop. No GPU needed (XGBoost `hist` on CPU; SVC is CPU-only).

Local run stopped 2026-09-22 after 8/30 XGB trials (best mean CV F1 ~0.246) to free
the laptop. Treat that as a smoke signal only — real metrics come from the finished
Colab run (`models/final_metrics.json`).

## Steps

1. **Push this repo to GitHub** (it is already `git init` + committed locally;
   remote already points at https://github.com/Shanmukh-Villuri/loan-approval-predictor):
   ```powershell
   # on the laptop, inside loan-approval-predictor/
   git push -u origin main
   ```
   Do NOT commit `data/*.csv`, `models/*.pkl`, or `.venv/` (already gitignored).

2. **Open Colab**: https://colab.research.google.com → File → Open notebook → GitHub
   tab → paste your repo URL → open `notebooks/02_colab_full_training.ipynb`.
   (Alternative without GitHub: upload the repo zip + the notebook via the Files panel.)
   Runtime → Change runtime type → **CPU + High-RAM** if available.

3. **Data** (pick one, in-notebook cells 3a / 3b):
   - **3a (recommended):** upload `kaggle.json` (from https://www.kaggle.com/settings →
     Account → API → Create New Token). The notebook downloads `application_train.csv`
     via the Kaggle API — exact provenance, matches the README source link.
   - **3b (fallback):** manually upload `application_train.csv` (~166 MB) from your laptop.

4. **Run all cells.** Cell 4 verifies the expected fingerprint
   (307511×122, ~8.07% minority, 260 encoded features) before the expensive steps.
   Cell 5 trains (hours — keep the tab open; Colab may recycle idle runtimes).
   To validate quickly first, run the smoke test from cell 5's comment:
   `!python -m src.train --quick --out models_quick`.

5. **Bring results home**: cell 7 downloads `final_metrics.json`,
   `training_summary.json`, and `reports/figures/*.png`. On the laptop, copy them into
   `models/` and `reports/figures/`, paste the ACTUAL numbers into the README metrics
   table (replacing PENDING), and commit:
   ```powershell
   git add models/final_metrics.json models/training_summary.json reports/figures README.md
   git commit -m "Real Colab metrics: <one-line summary>"
   ```

## If pinned installs fail

Colab's default Python may be newer than 3.11, in which case `numpy==1.26.4` (pinned for
our `.python-version`) can fail to install. Fallback: install compatible latest versions
instead of `-r requirements.txt`:
```
%pip install -q scikit-learn xgboost imbalanced-learn optuna pandas matplotlib seaborn shap streamlit joblib
```
then re-run. Note the substitution honestly in the README Results section (versions are
recorded in `training_summary.json` either way).

## Cost / time expectations

- Free-tier Colab CPU is enough but slow; a few hours for the full run is normal.
- Most time goes to Optuna trials × 3-fold CV with SMOTE + SVD inside each fold.
- To cut time honestly (document any change): lower `--n-trials-xgb/--n-trials-rf/--n-trials-svc`
  or raise `--svc-sample-frac` down / `--cv-folds` to 2. Never touch the test set.
