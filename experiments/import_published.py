"""Converts the per-fold HDLSS results published by TabPFN-Wide into our results format (the CSV of
eval_benchmark.py), so analyze_benchmark.py can put them next to ours. A dataset is imported only if our
folds are identical to theirs (same test labels in the same order, fold by fold); otherwise it is reported
and left out. Models are labelled "paper/<model>", e.g. "paper/wide-v2-5k".

Example, from the repo root:
    python experiments/import_published.py --tabpfn-wide-root ../TabPFN-Wide \
        --models v2 wide-v2-5k tabicl random_forest xgboost realmlp --output workdir/eval/hdlss_published.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from tfmplayground.benchmarks import hdlss
from tfmplayground.benchmarks.runner import RESULT_COLUMNS

REPO = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--tabpfn-wide-root", required=True, help="clone of TabPFN-Wide (has analysis_results/)")
parser.add_argument("--models", nargs="+", default=["v2", "wide-v2-5k", "tabicl", "random_forest", "xgboost", "realmlp"])
parser.add_argument("--data-root", default=str(REPO.parent), help="folder that contains HDLSS/")
parser.add_argument("--output", default="workdir/eval/hdlss_published.csv")
args = parser.parse_args()

published = {m: hdlss.published_results(args.tabpfn_wide_root, m) for m in args.models}
rows = []
for name in hdlss.DATASETS:
    X, y, _ = hdlss.load_task(name, args.data_root)
    for model, df in published.items():
        if not hdlss.check_folds_match(df, name, y):
            print(f"{name} | {model}: folds differ from ours, left out", flush=True)
            continue
        part = df[df.dataset_name == name]
        rows.append(pd.DataFrame({"benchmark": "hdlss", "dataset": name, "n_features": X.shape[1],
                                  "model": f"paper/{model}", "fold": part.fold.astype(int),
                                  "accuracy": part.accuracy, "roc_auc": part.roc_auc_score, "seconds": float("nan")}))
    print(f"{name} | {X.shape[0]} samples x {X.shape[1]} features | {len(set(y))} classes | {len(hdlss.folds(y))} folds",
          flush=True)

if not rows:
    raise SystemExit("No dataset has folds identical to the published ones: nothing imported.")
out = pd.concat(rows, ignore_index=True)[RESULT_COLUMNS]
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.output, index=False)
print(f"\n{len(out)} rows from {out.model.nunique()} models on {out.dataset.nunique()} datasets -> {args.output}")
