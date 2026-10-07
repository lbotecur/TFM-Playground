"""HDLSS benchmark (high-dimensional, low sample size): the 15 scikit-feature datasets of TabPFN-Wide
(analysis/hdlss_benchmark.py), prepared exactly as they do, so that the folds are identical and results
can be compared fold by fold with their published per-fold results.

Data: the .mat files from https://jundongl.github.io/scikit-feature/datasets.html, expected at
<data_root>/HDLSS/<name>.mat.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.io import loadmat
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

FOLDER = "HDLSS"  # expected inside the data root
DATASETS = ("ALLAML", "BASEHOCK", "CLL_SUB_111", "GLI_85", "PCMAC", "Prostate_GE", "RELATHE", "SMK_CAN_187",
            "TOX_171", "arcene", "colon", "gisette", "leukemia", "lung", "madelon")
N_SPLITS, SEED, LARGE = 3, 42, 2500  # 10 repeats, or 3 for datasets with LARGE samples or more


def load_task(name: str, data_root: str | Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Features (float32), encoded labels and categorical columns (none, as TabPFN-Wide). Rows are shuffled
    with seed 42 before anything else, as they do: the folds below refer to this order."""
    data = loadmat(Path(data_root) / FOLDER / f"{name}.mat")
    X, y = data["X"], np.asarray(data["Y"]).ravel()
    if scipy.sparse.issparse(X):
        X = X.toarray()
    X, y = shuffle(X, y, random_state=SEED)
    return np.asarray(X, dtype=np.float32), LabelEncoder().fit_transform(y), []


def folds(y: np.ndarray):
    """The (train_idx, test_idx) pairs TabPFN-Wide uses: stratified 3-fold, 10 repeats (3 if the dataset
    has 2500 samples or more), seed 42."""
    repeats = 10 if len(y) < LARGE else 3
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=repeats, random_state=SEED)
    return list(cv.split(np.zeros(len(y)), y))


def published_results(tabpfn_wide_root: str | Path, model: str) -> pd.DataFrame:
    """Per-fold HDLSS results published by TabPFN-Wide for one model directory, e.g. "wide-v2-5k", "v2",
    "tabicl", "random_forest", "xgboost", "realmlp"."""
    return pd.read_csv(Path(tabpfn_wide_root) / "analysis_results" / model / "hdlss_benchmark_results.csv")


def check_folds_match(published: pd.DataFrame, dataset: str, y: np.ndarray) -> bool:
    """True if our test folds have exactly the same labels, in the same order, as the published ones
    (their 'ground_truth' column) for this dataset."""
    rows = published[published.dataset_name == dataset].sort_values("fold")
    ours = folds(y)
    if len(rows) != len(ours):
        return False
    for (_, test_idx), (_, row) in zip(ours, rows.iterrows()):
        if not np.array_equal(np.array(str(row.ground_truth).split(), dtype=float).astype(int), y[test_idx]):
            return False
    return True
