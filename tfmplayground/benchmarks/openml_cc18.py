"""OpenML-CC18 (Bischl et al., 2021; OpenML suite 99): curated classification datasets, here restricted to the
narrow, small tables our model and TabPFN were built for: at most 10,000 rows, 500 features and 10 classes.
madelon is left out (a development dataset of ours, used for decisions).

Data: written by experiments/download_openml_cc18.py to <data_root>/OpenML-CC18/<name>.npz (features as float32
with categorical columns ordinal-coded and NaN for missing values, encoded labels, categorical column indices),
with an index.csv listing them. Folds: stratified 10-fold, seed 42, the same for every model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

FOLDER = "OpenML-CC18"
SUITE_ID = 99
MAX_ROWS, MAX_FEATURES, MAX_CLASSES = 10_000, 500, 10
EXCLUDED = ("madelon",)  # development dataset (decisions were made on it)
N_SPLITS, SEED = 10, 42


def keep(name: str, rows: int, features: int, classes: int) -> bool:
    """Whether a CC18 dataset is in our narrow subset (features exclude the target)."""
    return name not in EXCLUDED and rows <= MAX_ROWS and features <= MAX_FEATURES and 2 <= classes <= MAX_CLASSES


def to_arrays(X: pd.DataFrame, y, categorical: list[bool]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """OpenML data -> float32 features (categorical columns as ordinal codes, missing values as NaN), labels
    encoded 0..C-1, and the indices of the categorical columns."""
    columns, cat_idx = [], []
    for j, column in enumerate(X.columns):
        values = X[column]
        if categorical[j] or not pd.api.types.is_numeric_dtype(values):
            codes = pd.Categorical(values).codes.astype(np.float32)
            columns.append(np.where(codes < 0, np.nan, codes))  # -1 = missing
            cat_idx.append(j)
        else:
            columns.append(pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32))
    return np.stack(columns, axis=1).astype(np.float32), LabelEncoder().fit_transform(np.asarray(y)), cat_idx


def save_task(path: str | Path, X: np.ndarray, y: np.ndarray, categorical: list[int]) -> None:
    np.savez_compressed(path, X=X, y=y, categorical=np.asarray(categorical, dtype=np.int64))


def load_task(name: str, data_root: str | Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    data = np.load(Path(data_root) / FOLDER / f"{name}.npz")
    return data["X"].astype(np.float32), data["y"].astype(np.int64), data["categorical"].tolist()


def folds(y: np.ndarray):
    """Stratified 10-fold, seed 42."""
    return list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def datasets(data_root: str | Path) -> list[str]:
    """Names of the downloaded datasets, from index.csv."""
    return pd.read_csv(Path(data_root) / FOLDER / "index.csv").name.tolist()
