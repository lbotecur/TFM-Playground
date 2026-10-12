"""Do the models find the informative features, and do they predict well? The synthetic benchmark of TabPFN-Wide
(Appendix F): make_classification with 50 training samples, 200-5000 features of which k in {3, 5, 7} are
informative (no redundant or repeated ones, columns in random order), 5 seeds; plus held-out test samples from the
same generator (200 by default), so that each model's importance is reported with its predictive AUROC and
accuracy on unseen rows. Importance metrics: Recall@k (share of the informative features among the k top-scored)
and the AUROC of the scores between informative and noise features. Resumable: rows already in the CSV are skipped.

From the repo root (TabPFN-Wide needs tfm_ext):
    python experiments/interpretability_synthetic.py --gpu 0 --output workdir/eval/interp_synthetic_pred.csv \\
        --models workdir/release/nanotabpfn_v2.pth workdir/graph_scm_base_w5000/epoch_100.pth \\
                 tabpfn-wide-5k random_forest
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tfmplayground.interpretability import fit_and_explain, informative_auroc, recall_at_k

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", required=True)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--features", nargs="+", type=int, default=[200, 500, 1000, 2000, 5000])
parser.add_argument("--informative", nargs="+", type=int, default=[3, 5, 7])
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
parser.add_argument("--samples", type=int, default=50, help="training samples")
parser.add_argument("--test-samples", type=int, default=200)
parser.add_argument("--output", default="workdir/eval/interp_synthetic_pred.csv")
args = parser.parse_args()

os.chdir(REPO)
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
output = Path(args.output)
results = pd.read_csv(output) if output.exists() else pd.DataFrame()
done = set(map(tuple, results[["model", "features", "informative", "seed"]].values)) if len(results) else set()
for n_features in args.features:
    for k in args.informative:
        for seed in args.seeds:
            # informative features first, then the columns permuted: a fixed position would favour any model with
            # a bias for some columns (ours add a fixed random vector per column position)
            X, y = make_classification(n_samples=args.samples + args.test_samples, n_features=n_features,
                                       n_informative=k, n_redundant=0, n_repeated=0, shuffle=False, random_state=seed)
            order = np.random.default_rng(seed).permutation(n_features)
            X = X[:, order]
            informative = np.flatnonzero(order < k)
            # shuffle=False also leaves the rows sorted by class: a stratified random split
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=args.samples, stratify=y,
                                                                random_state=seed)
            for name in args.models:
                if (name, n_features, k, seed) in done:
                    continue
                start = time.time()
                scores, proba = fit_and_explain(name, X_train, y_train, X_test, device=device, seed=seed)
                row = {"model": name, "features": n_features, "informative": k, "seed": seed,
                       "recall_at_k": recall_at_k(scores, informative),
                       "informative_auroc": informative_auroc(scores, informative),
                       "test_auroc": roc_auc_score(y_test, proba[:, 1]),
                       "test_accuracy": accuracy_score(y_test, proba.argmax(axis=1)),
                       "seconds": round(time.time() - start, 1)}
                results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                output.parent.mkdir(parents=True, exist_ok=True)
                results.to_csv(output, index=False)
                print(f"{name} | p={n_features} k={k} seed={seed} | recall@k {row['recall_at_k']:.2f} | "
                      f"importance AUROC {row['informative_auroc']:.3f} | test AUROC {row['test_auroc']:.3f} | "
                      f"{row['seconds']:.0f}s", flush=True)

summary = results.groupby(["model", "features"])[["recall_at_k", "informative_auroc", "test_auroc"]].mean().round(3)
print(summary.to_string())
