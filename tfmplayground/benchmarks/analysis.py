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


# ---------------------------------------------------------------------------------------------------------
# Dataset-level analysis (one value per dataset and model): the unit the statistics below treat as independent.

LOW_MEMORY = " [low-memory]"


def merge_variants(results: pd.DataFrame, suffix: str = LOW_MEMORY) -> pd.DataFrame:
    """Treat "<label><suffix>" as the same model as "<label>" (low-memory inference is the same model and was
    validated equivalent). Where both exist for a fold, the row without the suffix is kept."""
    r = results.copy()
    r["_variant"] = r.model.str.endswith(suffix)
    r["model"] = r.model.str.replace(suffix, "", regex=False)
    r = r.sort_values("_variant", kind="stable").drop_duplicates(["model", *KEYS])
    return r.drop(columns="_variant").sort_index()


def short_name(label: str) -> str:
    """'workdir/graph_scm_base_w5000/epoch_100.pth [bfloat16]' -> 'graph_scm_base_w5000/epoch_100'."""
    label = label.split(" [")[0]
    if label.endswith(".pth"):
        parts = label[:-4].split("/")
        return "/".join(parts[-2:])
    return label


def omic(dataset: str) -> str:
    """'BRCA/mrna+cnv+methylation+mirna' -> 'mrna+cnv+methylation+mirna'; names without '/' map to themselves."""
    return dataset.split("/", 1)[-1]


def dataset_table(results: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """Mean over folds (NaN folds skipped) with the dataset as index, one column per model."""
    return results.groupby(["dataset", "model"])[metric].mean().unstack("model")


def mean_ranks(table: pd.DataFrame) -> pd.Series:
    """Average rank of each model over the rows (datasets) of table, 1 = best (highest), ties averaged.
    Only complete rows are used."""
    complete = table.dropna()
    return complete.rank(axis=1, ascending=False).mean().sort_values()


def friedman_nemenyi(table: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Friedman test over complete rows and the Nemenyi critical difference of mean ranks (Demsar, 2006):
    two models whose mean ranks differ by more than cd are significantly different at alpha."""
    from scipy.stats import friedmanchisquare, studentized_range

    complete = table.dropna()
    n, k = complete.shape
    stat, p = friedmanchisquare(*[complete[c].to_numpy() for c in complete.columns])
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return {"datasets": n, "models": k, "chi2": float(stat), "p": float(p),
            "cd": float(q * np.sqrt(k * (k + 1) / (6 * n)))}


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, same order as the input (NaN stays NaN)."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    order = valid[np.argsort(p[valid])]
    m, running = len(order), 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adjusted[idx] = running
    return adjusted


def against_reference(table: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Every model against reference with one pair per dataset (rows where both are present): wins / ties /
    losses, mean and median difference, Wilcoxon p-value and Holm-adjusted p over the comparisons."""
    rows = []
    for model in table.columns.drop(reference):
        pair = table[[model, reference]].dropna()
        diff = (pair[model] - pair[reference]).to_numpy()
        nonzero = diff[diff != 0]
        rows.append({"model": model, "datasets": len(diff), "wins": int((diff > 0).sum()),
                     "ties": int((diff == 0).sum()), "losses": int((diff < 0).sum()),
                     "mean_diff": float(diff.mean()) if len(diff) else np.nan,
                     "median_diff": float(np.median(diff)) if len(diff) else np.nan,
                     "wilcoxon_p": float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 2 else np.nan})
    out = pd.DataFrame(rows)
    out["holm_p"] = holm(out.wilcoxon_p)
    return out.sort_values("mean_diff", ascending=False, ignore_index=True)


def by_group(table: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Mean of table over the datasets of each group (e.g. groups = table.index.map(omic))."""
    return table.groupby(groups).mean()


def seconds_table(results: pd.DataFrame, groups: pd.Series | None = None) -> pd.DataFrame:
    """Median seconds per fold for each model, overall or per group of datasets (groups indexed by dataset)."""
    r = results.copy()
    r["group"] = r.dataset.map(groups) if groups is not None else "all"
    return r.groupby(["group", "model"]).seconds.median().unstack("model")
