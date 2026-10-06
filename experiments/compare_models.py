"""Tables from the per-fold CSV of eval_benchmark.py: mean AUROC per dataset and model, and every model
against a reference on the folds both have (wins / ties / losses, mean difference, Wilcoxon).

Example, from the repo root:
    python experiments/compare_models.py --reference "workdir/graph_scm_base_w5000/epoch_100.pth [bfloat16]"
"""

import argparse

import pandas as pd

from tfmplayground.benchmarks.analysis import mean_table, paired

parser = argparse.ArgumentParser()
parser.add_argument("--results", nargs="+", default=["workdir/eval/benchmarks.csv"], help="one or more CSVs")
parser.add_argument("--reference", required=True, help="model label as written in the CSV")
parser.add_argument("--benchmark", default=None, help="only this benchmark")
parser.add_argument("--metric", default="roc_auc")
args = parser.parse_args()

results = pd.concat([pd.read_csv(f) for f in args.results], ignore_index=True)
if args.benchmark:
    results = results[results.benchmark == args.benchmark]
pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 60)

print(f"Mean {args.metric} over folds\n")
print(mean_table(results, args.metric).round(3).to_string())

print(f"\nEach model against {args.reference} (positive mean_diff = model better)\n")
rows = [paired(results, m, args.reference, args.metric) for m in sorted(results.model.unique()) if m != args.reference]
print(pd.DataFrame(rows).drop(columns="reference").round(4).to_string(index=False))
