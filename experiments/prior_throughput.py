"""Tables per second that the TabICL prior generates with each number of workers, on the CPU only, to decide how
many cores each training run needs. A run on one GPU consumed about 18 tables/s of up to 300 rows (0.44 s of
GPU per step of 8 tables), so it needs at least that many to never wait.

From the repo root, for example:
    python experiments/prior_throughput.py --workers 1 4 8 --max-rows 300
    python experiments/prior_throughput.py --workers 4 8 --max-rows 1024
"""

import argparse
import time

import torch

from tfmplayground.external_priors import TabICLPriorDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
parser.add_argument("--prior-type", default="graph_scm")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--min-rows", type=int, default=40)
parser.add_argument("--max-rows", type=int, default=300)
parser.add_argument("--min-features", type=int, default=2)
parser.add_argument("--max-features", type=int, default=100)
parser.add_argument("--steps", type=int, default=40, help="batches timed per setting")
parser.add_argument("--graph-fct-types", default=None)
args = parser.parse_args()

for workers in args.workers:
    loader = TabICLPriorDataLoader(
        num_steps=args.steps, batch_size=args.batch_size, num_datapoints_min=args.min_rows,
        num_datapoints_max=args.max_rows, min_features=args.min_features, max_features=args.max_features,
        max_num_classes=10, device=torch.device("cpu"), prior_type=args.prior_type, log_seq_len=True,
        min_train_size=0.3, max_train_size=0.9, num_workers=0 if workers == 1 else workers,
        graph_fct_types=args.graph_fct_types,
    )
    iterator = iter(loader)
    next(iterator)  # workers started and warm
    start, tables = time.time(), 0
    for batch in iterator:
        tables += batch["x"].shape[0]
    seconds = time.time() - start
    print(f"workers {workers:2d} | rows {args.min_rows}-{args.max_rows} | {tables / seconds:6.1f} tables/s | "
          f"{seconds / max(tables / args.batch_size, 1):.2f} s per step of {args.batch_size}", flush=True)
