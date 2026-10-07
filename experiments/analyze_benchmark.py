"""Dataset-level analysis of the per-fold CSVs written by eval_benchmark.py, for the paper.

Low-memory runs are merged with their model (see analysis.merge_variants). For each metric it prints:
mean over folds per dataset, means per omic, mean ranks, Friedman + Nemenyi critical difference, every model
against the reference with one pair per dataset (Wilcoxon, Holm), and median seconds per fold per omic.
--exclude drops datasets (e.g. a corrupted one) and the whole analysis is repeated without them.

Example, from the repo root:
    python experiments/analyze_benchmark.py --results workdir/eval/bench_*.csv \
        --reference graph_scm_base_w5000/epoch_100 --exclude LGG/methylation --output workdir/eval/analysis
"""

import argparse
from pathlib import Path

import pandas as pd

from tfmplayground.benchmarks.analysis import (against_reference, by_group, dataset_table, friedman_nemenyi,
                                               mean_ranks, merge_variants, omic, seconds_table, short_name)

parser = argparse.ArgumentParser()
parser.add_argument("--results", nargs="+", required=True, help="one or more per-fold CSVs")
parser.add_argument("--reference", required=True, help="short model name, e.g. graph_scm_base_w5000/epoch_100")
parser.add_argument("--benchmark", default=None, help="only this benchmark")
parser.add_argument("--metrics", nargs="+", default=["roc_auc", "accuracy"])
parser.add_argument("--exclude", nargs="*", default=[], help="datasets left out in the second pass")
parser.add_argument("--output", default=None, help="folder for the CSV tables (optional)")
args = parser.parse_args()

results = pd.concat([pd.read_csv(f) for f in args.results], ignore_index=True)
if args.benchmark:
    results = results[results.benchmark == args.benchmark]
results = merge_variants(results)
results["model"] = results.model.map(short_name)
pd.set_option("display.width", 250, "display.max_columns", 50)
out = Path(args.output) if args.output else None
if out:
    out.mkdir(parents=True, exist_ok=True)


def report(results: pd.DataFrame, tag: str) -> None:
    for metric in args.metrics:
        table = dataset_table(results, metric)
        groups = table.index.to_series().map(omic)
        print(f"\n===== {tag}: {metric} ({len(table)} datasets) =====\n")
        print(table.round(3).to_string())
        print(f"\nMean per omic\n\n{by_group(table, groups).round(3).to_string()}")
        print(f"\nOverall mean\n\n{table.mean().sort_values(ascending=False).round(4).to_string()}")
        print(f"\nMean rank (1 = best)\n\n{mean_ranks(table).round(2).to_string()}")
        print(f"\nFriedman + Nemenyi: {friedman_nemenyi(table)}")
        comparison = against_reference(table, args.reference)
        print(f"\nAgainst {args.reference}, one pair per dataset (positive = model better)\n")
        print(comparison.round(4).to_string(index=False))
        if out:
            table.to_csv(out / f"{tag}_{metric}_datasets.csv")
            comparison.to_csv(out / f"{tag}_{metric}_vs_reference.csv", index=False)
    groups = pd.Series({d: omic(d) for d in results.dataset.unique()})
    seconds = seconds_table(results, groups).T
    print(f"\n===== {tag}: median seconds per fold =====\n\n{seconds.round(1).to_string()}")
    if out:
        seconds.to_csv(out / f"{tag}_seconds.csv")


report(results, "all")
if args.exclude:
    report(results[~results.dataset.isin(args.exclude)], "excluded")
