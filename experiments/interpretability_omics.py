"""Interpretability on real omics data: MLOmics mRNA (BRCA, COAD, GBM, LGG, OV), the benchmark's 5 folds.

1. Ranking. Each selector ranks the genes with the training rows of the fold only: they are split 80/20
   (stratified); the model is fitted on the 80% and its attention read while it predicts the 20% (held-out rows,
   as in the synthetic benchmark). Test rows and test labels are never used. Selectors: our checkpoints and
   TabPFN-Wide (attention of the label to each gene), a random forest (impurity) and the ANOVA F-test (the classic
   univariate filter). The columns are put in a random order per fold (scores mapped back), so that stability
   across folds cannot come from a bias for column positions. Scores are saved per dataset, selector and fold.
2. Utility. An evaluator fitted on the fold's training rows restricted to the k top-ranked genes predicts the
   test rows (AUROC). References: all genes, and k random genes (5 draws). Evaluators: logistic regression (the
   same for every selector) and, optionally, checkpoints.
3. Summary: utility, stability of the top 100 across folds (Jaccard) and agreement between selectors, and for
   BRCA the PAM50 genes among the top k (its labels are PAM50 subtypes) with a hypergeometric p-value.

Resumable: existing scores and utility rows are reused. From the repo root (TabPFN-Wide needs tfm_ext):
    python experiments/interpretability_omics.py --gpu 0 \\
        --selectors workdir/release/nanotabpfn_v2.pth tabpfn-wide-5k random_forest anova
"""

import argparse
import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tfmplayground.benchmarks import mlomics
from tfmplayground.benchmarks.metrics import roc_auc
from tfmplayground.interpretability import PAM50, fit_and_explain, gene_symbol, hits_and_enrichment, jaccard, top_k

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--selectors", nargs="+", required=True)
parser.add_argument("--evaluators", nargs="+", default=["logreg"], help="logreg and/or checkpoint paths")
parser.add_argument("--datasets", nargs="+", default=[f"{d}/mrna" for d in mlomics.DATASETS])
parser.add_argument("--ks", nargs="+", type=int, default=[10, 50, 100, 500])
parser.add_argument("--random-draws", type=int, default=5)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--data-root", default=str(REPO.parent))
parser.add_argument("--output-dir", default="workdir/eval/interp_omics")
args = parser.parse_args()

os.chdir(REPO)
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
out = Path(args.output_dir)
(out / "scores").mkdir(parents=True, exist_ok=True)


def label(name: str) -> str:
    return name.replace("workdir/", "").replace("/", "_").removesuffix(".pth")


def evaluator(name: str):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    from tfmplayground.interface import NanoTabPFNClassifier

    return NanoTabPFNClassifier(model=name, device=device, amp_dtype=torch.bfloat16, categorical_features=[],
                                infer_categorical=False)


def safe_auc(y_true, proba) -> float:
    try:
        return roc_auc(y_true, np.asarray(proba))
    except ValueError:  # a class absent from the test rows
        return float("nan")


utility_csv = out / "utility.csv"
utility = pd.read_csv(utility_csv) if utility_csv.exists() else pd.DataFrame()
done = set(map(tuple, utility[["dataset", "fold", "evaluator", "selector", "k"]].values)) if len(utility) else set()


def add(row):
    global utility
    utility = pd.concat([utility, pd.DataFrame([row])], ignore_index=True)
    utility.to_csv(utility_csv, index=False)


genes_of = {}
for dataset in args.datasets:
    X_df, y, _ = mlomics.load_task(dataset, args.data_root)
    genes_of[dataset] = [str(c) for c in X_df.columns]
    X = np.asarray(X_df, dtype=np.float32)
    for fold, (train_idx, test_idx) in enumerate(mlomics.folds(y)):
        X_tr, y_tr, X_te, y_te = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        # 1. rankings from the training rows only
        inner_fit, inner_query = train_test_split(np.arange(len(y_tr)), test_size=0.2, stratify=y_tr,
                                                  random_state=fold)
        scores = {}
        for name in args.selectors:
            path = out / "scores" / f"{dataset.replace('/', '_')}__{label(name)}__fold{fold}.npy"
            if not path.exists():
                start = time.time()
                # columns in a random order per fold, scores mapped back: our models add a fixed vector per column
                # position, so without this a positional bias would look like stability across folds
                order = np.random.default_rng(1000 + fold).permutation(X.shape[1])
                s_perm, _ = fit_and_explain(name, X_tr[inner_fit][:, order], y_tr[inner_fit],
                                            X_tr[inner_query][:, order], device=device, seed=fold)
                s = np.empty_like(s_perm)
                s[order] = s_perm
                np.save(path, s)
                print(f"{dataset} fold {fold} | ranked by {label(name)} | {time.time() - start:.0f}s", flush=True)
            scores[label(name)] = np.load(path)
        # 2. utility of the top-k genes for each evaluator
        rng = np.random.default_rng(fold)
        for ev in args.evaluators:
            key = (dataset, fold, label(ev), "all genes", X.shape[1])
            if key not in done:
                proba = evaluator(ev).fit(X_tr, y_tr).predict_proba(X_te)
                add({"dataset": dataset, "fold": fold, "evaluator": label(ev), "selector": "all genes",
                     "k": X.shape[1], "auroc": safe_auc(y_te, proba)})
            for k in args.ks:
                for sel, s in scores.items():
                    if (dataset, fold, label(ev), sel, k) in done:
                        continue
                    cols = top_k(s, k)
                    proba = evaluator(ev).fit(X_tr[:, cols], y_tr).predict_proba(X_te[:, cols])
                    add({"dataset": dataset, "fold": fold, "evaluator": label(ev), "selector": sel, "k": k,
                         "auroc": safe_auc(y_te, proba)})
                if (dataset, fold, label(ev), "random", k) not in done:
                    aucs = []
                    for _ in range(args.random_draws):
                        cols = rng.choice(X.shape[1], k, replace=False)
                        proba = evaluator(ev).fit(X_tr[:, cols], y_tr).predict_proba(X_te[:, cols])
                        aucs.append(safe_auc(y_te, proba))
                    add({"dataset": dataset, "fold": fold, "evaluator": label(ev), "selector": "random", "k": k,
                         "auroc": float(np.nanmean(aucs))})
            print(f"{dataset} fold {fold} | utility with {label(ev)} done", flush=True)

# 3. summary
pd.set_option("display.width", 250)
selectors = [label(n) for n in args.selectors]
print("\n=== Utility: AUROC on the test rows with the top-k genes (mean over folds and datasets)")
print(utility.groupby(["evaluator", "k", "selector"]).auroc.mean().unstack().round(3).to_string())

rows = []
for dataset in args.datasets:
    folds = range(len(mlomics.folds(mlomics.load_task(dataset, args.data_root)[1])))
    loaded = {(sel, f): np.load(out / "scores" / f"{dataset.replace('/', '_')}__{sel}__fold{f}.npy")
              for sel in selectors for f in folds}
    for sel in selectors:
        tops = [top_k(loaded[sel, f], 100) for f in folds]
        rows.append({"dataset": dataset, "what": f"stability {sel}",
                     "jaccard_top100": np.mean([jaccard(a, b) for a, b in itertools.combinations(tops, 2)]),
                     "spearman": np.nanmean([spearmanr(loaded[sel, a], loaded[sel, b], nan_policy="omit").statistic
                                             for a, b in itertools.combinations(folds, 2)])})
    for a, b in itertools.combinations(selectors, 2):
        rows.append({"dataset": dataset, "what": f"agreement {a} / {b}",
                     "jaccard_top100": np.mean([jaccard(top_k(loaded[a, f], 100), top_k(loaded[b, f], 100))
                                                for f in folds]),
                     "spearman": np.nanmean([spearmanr(loaded[a, f], loaded[b, f], nan_policy="omit").statistic
                                             for f in folds])})
print("\n=== Stability across folds and agreement between selectors (top 100 genes; Spearman over all genes)")
print(pd.DataFrame(rows).groupby("what")[["jaccard_top100", "spearman"]].mean().round(3).to_string())
print(pd.DataFrame(rows).round(3).to_string())

from scipy.stats import hypergeom  # noqa: E402


def mean_scores(dataset: str, sel: str) -> np.ndarray:
    return np.mean([np.nan_to_num(np.load(out / "scores" / f"{dataset.replace('/', '_')}__{sel}__fold{f}.npy"),
                                  nan=-1.0) for f in range(5)], axis=0)


for dataset in args.datasets:
    symbols = [gene_symbol(g) for g in genes_of[dataset]]
    print(f"\n=== {dataset}: top 10 genes (mean score over the 5 folds)")
    for sel in selectors:
        print(f"{sel:45s} {', '.join(symbols[i] for i in top_k(mean_scores(dataset, sel), 10))}")
    if not dataset.startswith("BRCA"):
        continue
    universe, pam50 = set(symbols), set(PAM50) & set(symbols)
    print(f"PAM50 genes among the top k ({len(pam50)} of 50 present among {len(universe)} genes; mean over folds, "
          f"p = chance of at least that many in k random genes)")
    for sel in selectors:
        line = []
        for k in (10, 50, 100):
            hits = np.mean([hits_and_enrichment([symbols[i] for i in top_k(np.load(
                out / "scores" / f"{dataset.replace('/', '_')}__{sel}__fold{f}.npy"), k)], pam50, universe, k)[0]
                for f in range(5)])
            p_value = hypergeom.sf(round(hits) - 1, len(universe), len(pam50), k)
            line.append(f"top{k}: {hits:.1f} (p={p_value:.1e})")
        print(f"{sel:45s} " + " | ".join(line))
