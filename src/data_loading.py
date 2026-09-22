"""Loading utilities for Home Credit Default Risk (application_train.csv only).

Dataset source (exact):
    https://www.kaggle.com/competitions/home-credit-default-risk/data
    File: application_train.csv (~307k rows, ~122 raw columns, TARGET imbalance ~8%)

Only application_train.csv is used — no bureau / previous_application merges,
per project spec. The raw CSV is never committed (see .gitignore).

Download via Kaggle API (requires free Kaggle account + API token):
    kaggle competitions download -c home-credit-default-risk -f application_train.csv -p data/
    python -c "import zipfile; zipfile.ZipFile('data/application_train.csv.zip').extractall('data/')"
Or download manually from the link above and place the file at data/application_train.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "application_train.csv"


def kaggle_download_cmd(data_dir: str = "data/") -> str:
    """Return the Kaggle CLI command string for reproducing the download."""
    return (
        f"kaggle competitions download -c home-credit-default-risk "
        f"-f application_train.csv -p {data_dir}"
    )


def load_application_train(
    data_path: str | Path = DEFAULT_DATA_PATH,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load application_train.csv.

    Args:
        data_path: Path to application_train.csv.
        nrows: Optional row cap for smoke tests / low-memory runs.

    Returns:
        Raw dataframe including TARGET and SK_ID_CURR.

    Raises:
        FileNotFoundError: With actionable download instructions.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            f"Download it (free Kaggle account required):\n"
            f"  1. Visit https://www.kaggle.com/competitions/home-credit-default-risk/data\n"
            f"  2. Download application_train.csv into data/\n"
            f"  Or via CLI: {kaggle_download_cmd()}\n"
            f"  Then unzip into data/application_train.csv"
        )
    df = pd.read_csv(path, nrows=nrows)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected column {TARGET_COL!r} in {path}")
    return df


def make_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    target_col: str = TARGET_COL,
    id_col: str = ID_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split; drops ID col from features.

    Args:
        df: Full dataframe with target column.
        test_size: Held-out fraction.
        random_state: Seed for reproducibility.
        target_col: Name of label column.
        id_col: Row identifier to drop from X.

    Returns:
        (X_train, X_test, y_train, y_test). Test set is never resampled.
    """
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col, id_col], errors="ignore")
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def minority_rate(y: pd.Series) -> float:
    """Return fraction of positive (high-risk) class."""
    return float((y == 1).mean())
