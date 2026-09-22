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

> **If cell 0 says the CSV isn't mounted:** you skipped step 2. Right sidebar →
> **+ Add Data → Competition** (not Dataset) → search `Home Credit Default Risk` →
> **Add**, then re-run cell 0. Cell 0 lists what's actually mounted and auto-detects
> the file if it arrived under a different folder name.

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
   Record the headline numbers in the README if you want them there.
   If any resume-draft target is missed, say so in Results with reasons.

## Dependencies on Kaggle (important)

Do NOT `pip install -r requirements.txt` on Kaggle. Those pins target our local
Python 3.11.9 env; on the Kaggle image (Python 3.12+) `numpy==1.26.4` downgrades the
preinstalled numpy 2.x and corrupts scipy/sklearn (`No module named 'numpy.char'`).
The notebook therefore installs only what's missing from the image
(`xgboost imbalanced-learn optuna shap`) and keeps the image's numpy/scipy/sklearn/pandas.
`training_summary.json` records the versions actually used — cite those, not the pins.

**Recovering an env already broken by the pinned install:** run once, then
Kernel → Restart, then continue from the (fixed) install cell:
```
!pip install -q --upgrade --force-reinstall numpy scipy scikit-learn pandas xgboost imbalanced-learn optuna shap joblib matplotlib seaborn
```
This forces one consistent latest set. Afterwards the minimal install line is a no-op.

## Time expectations

Multi-hour full run is normal (Optuna trials × 3-fold CV, SMOTE+SVD inside each fold).
To shorten honestly — and document it — lower `--n-trials-xgb/--n-trials-rf/--n-trials-svc`
or use `--cv-folds 2`. Never touch the held-out test set; SMOTE stays inside folds.
