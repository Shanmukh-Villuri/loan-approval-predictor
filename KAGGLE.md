# Training on Kaggle Notebooks (recommended — data mounts directly)

Colab proved flaky (uploads + long runs + disconnects). Kaggle Notebooks remove the
data problem: attach the competition dataset and it appears at
`/kaggle/input/home-credit-default-risk/application_train.csv`. No `kaggle.json`,
no 166 MB upload.

## Steps

1. **Kaggle → Create → Notebook** (or File → Import Notebook and upload
   `notebooks/03_kaggle_full_training.ipynb` from this repo).

2. **Attach data**: in the right sidebar, **+ Add Data → Competition →
   `Home Credit Default Risk` → Add**. Verify the input path exists
   (notebook cell 0 lists it and asserts).

3. **Settings** (right panel): **Internet ON** (needed for `git clone` + `pip install`),
   Accelerator **CPU**. Persistence isn't needed for the run itself — outputs land in
   `/kaggle/working/`, which committed versions preserve.

4. **Paste / import** the cells from `notebooks/03_kaggle_full_training.ipynb`
   (already prefilled with this repo's URL):
   - cell 1 clones https://github.com/Shanmukh-Villuri/loan-approval-predictor
     and installs `requirements.txt` (pinned for 3.11),
   - cell 2 fingerprints the data (expect 307511×122, ~8.07% minority, 260 features),
   - cell 3 trains (`src.train`, Optuna 30/30/20, SVD-50 benchmark, SMOTE-in-folds, stack),
   - cell 4 evaluates (`src.evaluate`) and prints `final_metrics.json`.

5. **Execute headless**: **Save Version → Save & Run All (Commit)** — not interactive
   Run All. Committed runs execute up to 9h in the background with versioned outputs,
   immune to closed tabs. Optional: run the `--quick` smoke test interactively first
   (see cell 3's comment) to validate the pipeline in minutes.

6. **Bring results home**: open the finished version → **Output** tab → download
   `models/final_metrics.json`, `models/training_summary.json`, and
   `reports/figures/*.png`. On the laptop:
   ```powershell
   # copy downloads into the repo, then:
   git add models/final_metrics.json models/training_summary.json reports/figures README.md
   git commit -m "Real Kaggle metrics: <one-line summary>"
   ```
   Paste the ACTUAL numbers into the README metrics table (replacing PENDING).
   If any resume-draft target is missed, say so in Results with reasons.

## If pinned installs fail

Kaggle images move Python versions over time. If `numpy==1.26.4` (pinned for our
`.python-version` 3.11.9) refuses to install, fall back to latest compatibles:
```
!pip install -q scikit-learn xgboost imbalanced-learn optuna pandas matplotlib seaborn shap streamlit joblib
```
Note the substitution in the README Results section. Either way,
`training_summary.json` records the versions actually used.

## Time expectations

Multi-hour full run is normal (Optuna trials × 3-fold CV, SMOTE+SVD inside each fold).
To shorten honestly — and document it — lower `--n-trials-xgb/--n-trials-rf/--n-trials-svc`
or use `--cv-folds 2`. Never touch the held-out test set; SMOTE stays inside folds.
