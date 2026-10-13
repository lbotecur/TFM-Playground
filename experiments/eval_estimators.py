"""How many ensemble passes (TabOmics' n_estimators) are worth it: AUROC, accuracy and time on the development
datasets, with the same folds as validate_run.py. Decisions are taken on these datasets only, never on the paper
benchmarks. Resumable (rows already in the CSV are skipped). Needs TabOmics installed (pip install -e ../tabomics
--no-deps) on its feat/estimators branch or later. From the repo root:

    nohup python -u experiments/eval_estimators.py --gpu 2 --model workdir/release/tabomics-v2 \\
        --n-estimators 1 4 8 16 > workdir/eval/estimators_dev.log 2>&1 &
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from tabomics import Classifier

from tfmplayground.benchmarks import hdlss
from tfmplayground.benchmarks.metrics import accuracy, roc_auc
from tfmplayground.validation import (
    BENCHMARK_DEV_DATASETS,
    DEV_DATASETS,
    DEV_FOLDER,
    dev_folds,
    load_dev_dataset,
)

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="TabOmics release directory, .pth or published name (v2)")
parser.add_argument("--n-estimators", type=int, nargs="+", default=[1, 4, 8, 16])
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--data-root", default=str(REPO.parent))
parser.add_argument("--output", default="workdir/eval/estimators_dev.csv")
args = parser.parse_args()

output = REPO / args.output
output.parent.mkdir(parents=True, exist_ok=True)
results = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=["dataset", "fold", "n_estimators"])
done = set(map(tuple, results[["dataset", "fold", "n_estimators"]].values))

for dataset in DEV_DATASETS + BENCHMARK_DEV_DATASETS:
    if dataset in BENCHMARK_DEV_DATASETS:
        if not (Path(args.data_root) / hdlss.FOLDER / f"{dataset}.mat").exists():
            continue
        X, y, _ = hdlss.load_task(dataset, args.data_root)
    elif (Path(args.data_root) / DEV_FOLDER / f"{dataset}.mat").exists():
        X, y = load_dev_dataset(dataset, args.data_root)
    else:
        continue
    for fold, (train, test) in enumerate(dev_folds(y)):
        for n in args.n_estimators:
            if (dataset, fold, n) in done:
                continue
            start = time.time()
            clf = Classifier(model=args.model, device=f"cuda:{args.gpu}", amp_dtype=torch.bfloat16,
                             categorical_features=[], infer_categorical=False, n_estimators=n)
            proba = clf.fit(X[train], y[train]).predict_proba(X[test])
            row = {"dataset": dataset, "fold": fold, "n_estimators": n, "roc_auc": roc_auc(y[test], proba),
                   "accuracy": accuracy(y[test], proba), "seconds": round(time.time() - start, 1)}
            results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
            results.to_csv(output, index=False)
            print(f"{dataset} fold {fold} n={n}: AUROC {row['roc_auc']:.4f} ({row['seconds']}s)", flush=True)

summary = results.groupby(["dataset", "n_estimators"])[["roc_auc", "accuracy", "seconds"]].mean().unstack()
print(summary.round(4).to_string())
print(results.groupby("n_estimators")[["roc_auc", "accuracy", "seconds"]].mean().round(4).to_string())
