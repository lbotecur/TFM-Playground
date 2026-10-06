"""Runs models on a benchmark fold by fold and appends the results to a CSV, so that an interrupted
evaluation can be resumed and every fold can later be compared with published per-fold results."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from tfmplayground.benchmarks import mlomics
from tfmplayground.benchmarks.metrics import accuracy, roc_auc
from tfmplayground.benchmarks.models import is_checkpoint, make_model, predict_proba

COLUMNS = ["dataset_name", "omic", "checkpoint", "n_features", "fold", "accuracy", "roc_auc"]


def run_mlomics(
    models: list[str],
    mlomics_root: str | Path,
    output: str | Path,
    datasets: tuple[str, ...] = mlomics.DATASETS,
    omics: tuple[str, ...] = ("mrna",),
    n_features: tuple[int, ...] = (0,),
    device: str = "cuda",
    tabpfn_wide_root: str | Path | None = None,
    amp_dtype=None,
) -> pd.DataFrame:
    """Evaluates every model on every dataset and feature count with the TabPFN-Wide protocol.
    If tabpfn_wide_root is given, first checks that our test folds are identical to theirs.
    amp_dtype (e.g. torch.bfloat16) runs our checkpoints in mixed precision; those rows are labelled
    with the dtype (e.g. "path [bfloat16]") so that results of different precisions never mix."""
    output = Path(output)
    results = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=COLUMNS)
    omic_name = "+".join(omics)
    published = None
    if tabpfn_wide_root is not None and omics == ("mrna",):
        published = mlomics.published_results(tabpfn_wide_root, "random_forest")

    for dataset in datasets:
        X_full, y = mlomics.load(dataset, mlomics_root, omics)
        for n in n_features:
            if n > X_full.shape[1]:
                print(f"{dataset}: skipping {n} features, only {X_full.shape[1]} available", flush=True)
                continue
            X = mlomics.reduce_features(X_full, n)
            folds = mlomics.folds(y)
            if published is not None:
                same = mlomics.check_folds_match(published, dataset, X.shape[1], y)
                print(f"{dataset} | {X.shape[1]} features | folds identical to TabPFN-Wide: {'yes' if same else 'NO'}", flush=True)
            for model_name in models:
                ours = is_checkpoint(model_name)  # amp_dtype only applies to our checkpoints
                name = model_name
                if ours and amp_dtype is not None:
                    name = f"{model_name} [{str(amp_dtype).split('.')[-1]}]"
                done = set(
                    results[
                        (results.dataset_name == dataset) & (results.omic == omic_name)
                        & (results.checkpoint == name) & (results.n_features == X.shape[1])
                    ].fold
                )
                for i, (train_idx, test_idx) in enumerate(folds):
                    if i in done:
                        continue
                    model = make_model(model_name, device, amp_dtype if ours else None)
                    proba = predict_proba(model, X[train_idx], y[train_idx], X[test_idx])
                    row = {
                        "dataset_name": dataset, "omic": omic_name, "checkpoint": name, "n_features": X.shape[1],
                        "fold": i, "accuracy": accuracy(y[test_idx], proba), "roc_auc": roc_auc(y[test_idx], proba),
                    }
                    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                    print(f"{dataset} | {X.shape[1]} | {name} | fold {i} | AUROC {row['roc_auc']:.3f}", flush=True)
                output.parent.mkdir(parents=True, exist_ok=True)
                results.to_csv(output, index=False)
    return results


def compare(ours: pd.DataFrame, published: dict[str, pd.DataFrame], metric: str = "roc_auc") -> pd.DataFrame:
    """Mean and std over folds, per dataset and feature count, of our models and the published ones,
    side by side. Rows are matched on the actual number of features."""
    frames = [ours[["dataset_name", "checkpoint", "n_features", "fold", metric]]]
    for name, df in published.items():
        df = df[df.omic.str.lower() == "mrna"][["dataset_name", "n_features", "fold", metric]].copy()
        df["checkpoint"] = f"[TabPFN-Wide] {name}"
        frames.append(df)
    allres = pd.concat(frames, ignore_index=True)
    keys = set(map(tuple, ours[["dataset_name", "n_features"]].drop_duplicates().values))
    allres = allres[[k in keys for k in map(tuple, allres[["dataset_name", "n_features"]].values)]]
    stats = allres.groupby(["dataset_name", "n_features", "checkpoint"])[metric].agg(["mean", "std", "count"])
    return stats.reset_index()


BENCHMARKS = {"mlomics": mlomics}  # each module: load_task(name, data_root) -> X, y, categorical; folds(y)
RESULT_COLUMNS = ["benchmark", "dataset", "n_features", "model", "fold", "accuracy", "roc_auc", "seconds"]


def run_benchmark(
    datasets: list[str],
    models: list[str],
    output: str | Path,
    data_root: str | Path,
    n_features: tuple[int, ...] = (0,),
    device: str = "cuda",
    amp_dtype=None,
    low_memory: bool = False,
) -> pd.DataFrame:
    """Evaluates every model on every dataset ("<benchmark>/<name>", e.g. "mlomics/BRCA/mrna+cnv") and
    feature count (0 = all; otherwise feature agglomeration, only for all-numeric data), with the folds
    of each benchmark. Appends one row per fold to the CSV and skips folds already there, so it can be
    resumed. Every model is told which columns are categorical. Our checkpoints run in amp_dtype if
    given, and are then labelled with it (e.g. "path [bfloat16]"); with low_memory they are labelled
    "[low-memory]" too, so both versions can be compared fold by fold."""
    output = Path(output)
    results = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=RESULT_COLUMNS)
    for spec in datasets:
        benchmark, _, name = spec.partition("/")
        module = BENCHMARKS[benchmark]
        X_full, y, categorical = module.load_task(name, data_root)
        for n in n_features:
            if n > X_full.shape[1]:
                print(f"{spec}: skipping {n} features, only {X_full.shape[1]} available", flush=True)
                continue
            if n and categorical:
                raise ValueError(f"{spec}: feature agglomeration needs all-numeric data")
            X = mlomics.reduce_features(X_full, n) if n else np.asarray(X_full, dtype=np.float32)
            folds = module.folds(y)
            for model_name in models:
                ours = is_checkpoint(model_name)
                label = model_name
                if ours and amp_dtype is not None:
                    label = f"{model_name} [{str(amp_dtype).split('.')[-1]}]"
                if ours and low_memory:
                    label = f"{label} [low-memory]"
                done = set(results[(results.benchmark == benchmark) & (results.dataset == name)
                                   & (results.n_features == X.shape[1]) & (results.model == label)].fold)
                for i, (train_idx, test_idx) in enumerate(folds):
                    if i in done:
                        continue
                    start = time.time()
                    model = make_model(model_name, device, amp_dtype if ours else None, categorical=categorical,
                                       low_memory=low_memory and ours)
                    proba = predict_proba(model, X[train_idx], y[train_idx], X[test_idx])
                    row = {"benchmark": benchmark, "dataset": name, "n_features": X.shape[1], "model": label,
                           "fold": i, "accuracy": accuracy(y[test_idx], proba), "roc_auc": roc_auc(y[test_idx], proba),
                           "seconds": round(time.time() - start, 1)}
                    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                    print(f"{spec} | {X.shape[1]} | {label} | fold {i} | AUROC {row['roc_auc']:.3f} | "
                          f"{row['seconds']:.0f}s", flush=True)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    results.to_csv(output, index=False)  # after every fold: nothing is lost if it stops
    return results
