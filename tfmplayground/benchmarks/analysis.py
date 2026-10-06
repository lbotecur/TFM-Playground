"""Summaries of the per-fold results written by runner.run_benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEYS = ["benchmark", "dataset", "n_features", "fold"]


def mean_table(results: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """Mean over folds: one row per benchmark, dataset and feature count, one column per model."""
    return results.pivot_table(index=["benchmark", "dataset", "n_features"], columns="model", values=metric,
                               aggfunc="mean")


def paired(results: pd.DataFrame, model: str, reference: str, metric: str = "roc_auc") -> dict:
    """model against reference on the folds both have: number of folds, wins / ties / losses of model,
    mean difference and two-sided Wilcoxon p-value (folds of one dataset share samples, so they are not
    independent: take the p-value as indicative). Folds where either metric is NaN are left out (e.g. AUROC
    of a test fold that lacks a class, as COAD fold 0, also NaN in TabPFN-Wide's published results)."""
    a = results[results.model == model].set_index(KEYS)[metric]
    b = results[results.model == reference].set_index(KEYS)[metric]
    a, b = a.align(b, join="inner")
    both = a.notna() & b.notna()
    a, b = a[both], b[both]
    diff = (a - b).to_numpy()
    nonzero = diff[diff != 0]
    return {
        "model": model, "reference": reference, "folds": len(diff),
        "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()), "losses": int((diff < 0).sum()),
        "mean_diff": float(diff.mean()) if len(diff) else np.nan,
        "wilcoxon_p": float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 2 else np.nan,
    }
