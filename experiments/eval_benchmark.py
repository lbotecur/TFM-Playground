"""Evaluates models on benchmark datasets and appends per-fold results to one CSV (resumable).

Datasets are "<benchmark>/<name>": mlomics/<BRCA|COAD|GBM|LGG|OV>/<omics joined with +>, hdlss/<name>
(the 15 scikit-feature datasets in tfmplayground.benchmarks.hdlss.DATASETS), or shamir/<breast|gbm|sarcoma>/<omics
joined with +, from mrna, methylation, mirna>, or openml/<name> (the narrow OpenML-CC18 subset written by
experiments/download_openml_cc18.py; openml/all for all of them).
Models: our checkpoints (paths), random_forest, logreg, logreg_en, xgboost, tabpfn-wide-5k,
tabpfn-v2-gn2p4bpt, tabpfn-3.5 (":n8" = 8 ensemble members, ":auto" = package default, 8 members fixed by
the checkpoint, ":cover" = enough members, ceil(features / 768) and at least 8, for every feature to be seen).
External models need tfm_ext.
Our checkpoints run in low-memory inference by default (labelled [low-memory]; --no-low-memory for the normal
path), with the feature clip saved in their checkpoint (10 for v2 on, 100 for older ones); --feature-clip
overrides it (labelled e.g. [clip10]).

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
parser.add_argument("--low-memory", action=argparse.BooleanOptionalAction, default=True,
                    help="our checkpoints: low-memory inference (default, labelled [low-memory]); --no-low-memory "
                         "uses the normal path except for tables too large for it, unlabelled, as earlier runs")
parser.add_argument("--feature-clip", type=float, default=None,
                    help="our checkpoints: bound of the normalized features (default: the checkpoint's own, 100 "
                         "for every checkpoint saved before it was configurable); labelled e.g. [clip10]")
parser.add_argument("--permute-columns", action=argparse.BooleanOptionalAction, default=None,
                    help="our checkpoints: shuffle the columns once with a fixed seed, or not (default: the "
                         "checkpoint's own setting, off for the v2 and earlier); labelled [permuted]/[not permuted]")
parser.add_argument("--output", default="workdir/eval/benchmarks.csv")
parser.add_argument("--data-root", default=str(REPO.parent), help="folder that contains each benchmark's data")
args = parser.parse_args()

os.chdir(REPO)
datasets = []
for spec in args.datasets:  # "openml/all": every downloaded dataset of the OpenML-CC18 subset
    if spec == "openml/all":
        from tfmplayground.benchmarks import openml_cc18

        datasets += [f"openml/{name}" for name in openml_cc18.datasets(args.data_root)]
    else:
        datasets.append(spec)
run_benchmark(
    datasets=datasets,
    models=args.models,
    output=args.output,
    data_root=args.data_root,
    n_features=tuple(args.n_features),
    device=f"cuda:{args.gpu}",
    amp_dtype=torch.bfloat16 if args.bf16 else None,
    low_memory=args.low_memory,
    feature_clip=args.feature_clip,
    permute_columns=args.permute_columns,
)
