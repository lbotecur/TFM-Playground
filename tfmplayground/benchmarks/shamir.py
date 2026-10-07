"""Shamir multi-omics benchmark (Rappoport & Shamir, 2018: TCGA mRNA, methylation and miRNA), with the subtype
labels TabPFN-Wide uses (analysis/load_mm_data.py, load_multiomics_benchmark_shamir with subtype_labels=True):
breast PAM50, gbm expression subtype and sarcoma histological type. Their colon labels come from a
'subtype_labels' file that is not part of the public download, and kidney has a single subtype, so both are
left out.

Prepared as they do: each omic is standardized over all samples, duplicated samples are dropped, and only
patients with all three omics and a label are kept. One difference: they intersect the samples through a
Python set, so their row order (and their folds) changes from run to run; here the samples are sorted, so the
folds are reproducible. Their per-fold results are not published, so every model is run here.

Data: https://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html, each zip unpacked in its own folder:
<data_root>/Shamir/<cancer>/{exp,methy,mirna} and <data_root>/Shamir/clinical/clinical/<cancer>.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, scale

FOLDER = "Shamir"  # expected inside the data root
LABEL_COLUMNS = {"breast": "PAM50Call_RNAseq", "gbm": "GeneExp_Subtype", "sarcoma": "histological_type"}
DATASETS = tuple(LABEL_COLUMNS)
OMIC_FILES = {"mrna": "exp", "methylation": "methy", "mirna": "mirna"}
N_SPLITS, N_REPEATS, SEED = 5, 1, 42


def _key(sample: str) -> str:
    """'TCGA.3C.AAAU.01' or 'TCGA-3C-AAAU-01' -> 'tcga.3c.aaau.01', to match omics and clinical samples."""
    return sample.replace("-", ".").lower()


def _load_omic(path: Path) -> pd.DataFrame:
    """One omic file (features x samples, whitespace separated) -> samples x features, standardized."""
    data = pd.read_table(path, header=0, index_col=0, sep=r"\s+").T
    data = data[~data.index.duplicated(keep="first")]
    assert not data.isna().any().any(), f"missing values in {path}"
    data[:] = scale(data)
    data.index = data.index.map(_key)
    return data


def load(dataset: str, root: str | Path, omics: tuple[str, ...]) -> tuple[pd.DataFrame, np.ndarray]:
    """Features (samples x features, omics concatenated in the given order) and encoded subtype labels of
    the patients that have every omic of the dataset and a label, sorted by sample ID."""
    assert dataset in DATASETS, f"{dataset} not in {DATASETS}"
    folder = Path(root) / FOLDER
    data = {omic: _load_omic(folder / dataset / file) for omic, file in OMIC_FILES.items()}
    clinical = pd.read_table(folder / "clinical" / "clinical" / dataset)
    labels = clinical.set_index(clinical["sampleID"].map(_key))[LABEL_COLUMNS[dataset]].dropna()
    labels = labels[~labels.index.duplicated(keep="first")]
    samples = sorted(set(labels.index).intersection(*(set(d.index) for d in data.values())))
    X = pd.concat([data[omic].loc[samples] for omic in omics], axis=1)
    return X, LabelEncoder().fit_transform(labels.loc[samples].to_numpy())


def load_task(name: str, data_root: str | Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """'breast/mrna' or 'breast/mrna+methylation+mirna' -> features, labels and categorical columns (none)."""
    dataset, _, omics = name.partition("/")
    X, y = load(dataset, data_root, tuple((omics or "mrna").split("+")))
    return X.to_numpy(dtype=np.float32), y, []


def folds(y: np.ndarray):
    """Stratified 5-fold, one repeat, seed 42, as TabPFN-Wide (analysis/data.py, generate_tensor_folds)."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    return list(cv.split(np.zeros(len(y)), y))
