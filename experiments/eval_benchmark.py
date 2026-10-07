"""Evaluates models on benchmark datasets and appends per-fold results to one CSV (resumable).

Datasets are "<benchmark>/<name>". For now: mlomics/<BRCA|COAD|GBM|LGG|OV>/<omics joined with +>.
Models: our checkpoints (paths), random_forest, logreg, logreg_en, xgboost, tabpfn-wide-5k,
tabpfn-v2-gn2p4bpt, tabpfn-3.5 (":n8" = 8 ensemble members, ":auto" = package default, 8 members fixed by
the checkpoint, ":cover" = enough members, ceil(features / 768) and at least 8, for every feature to be seen).
External models need tfm_ext.

Example, from the repo root:
    python experiments/eval_benchmark.py --gpu 0 --bf16 \\
        --datasets mlomics/BRCA/cnv mlomics/BRCA/mrna+cnv+methylation+mirna \\
        --models workdir/graph_scm_base_w5000/epoch_100.pth tabpfn-3.5 xgboost
"""

import argparse
import os
from pathlib import Path

import torch

from tfmplayground.benchmarks.runner import run_benchmark

REPO = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", required=True)
parser.add_argument("--models", nargs="+", required=True)
parser.add_argument("--n-features", nargs="+", type=int, default=[0], help="0 = all features")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--bf16", action="store_true", help="run our checkpoints in bf16 (labelled [bfloat16])")
parser.add_argument("--low-memory", action="store_true",
                    help="our checkpoints: low-memory inference for very wide tables (labelled [low-memory])")
parser.add_argument("--output", default="workdir/eval/benchmarks.csv")
parser.add_argument("--data-root", default=str(REPO.parent), help="folder that contains each benchmark's data")
args = parser.parse_args()

os.chdir(REPO)
run_benchmark(
    datasets=args.datasets,
    models=args.models,
    output=args.output,
    data_root=args.data_root,
    n_features=tuple(args.n_features),
    device=f"cuda:{args.gpu}",
    amp_dtype=torch.bfloat16 if args.bf16 else None,
    low_memory=args.low_memory,
)
