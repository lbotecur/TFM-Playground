"""How large a training table fits on one GPU: for each number of columns, the most rows whose training step
(forward and backward, as pretrain.py runs it: bf16, gradient checkpointing, one table per batch) stays under a
memory limit. The result sets the rows x columns budget of the shape sampling (rows shrink as columns grow).

From the repo root, on a free GPU (each shape is a short search, a few minutes in all):

    python -u experiments/memory_budget.py --gpu 3 > workdir/validation/memory_budget.log 2>&1

Prints and writes workdir/validation/memory_budget.csv: columns, largest rows that fit, peak memory (GB) and
seconds per step at that size, and rows x columns (the cells, whose maximum is the budget).
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import autocast

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--columns", type=int, nargs="+",
                    default=[100, 300, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000])
parser.add_argument("--max-rows", type=int, default=16384, help="largest rows tried")
parser.add_argument("--limit-gb", type=float, default=72.0,
                    help="peak memory allowed (GB); below the GPU's 80 so that real batches, with the prior's "
                         "variation and the allocator's fragmentation, keep a margin")
parser.add_argument("--train-share", type=float, default=0.7, help="share of training rows (the prior draws 0.3-0.9)")
parser.add_argument("--output", default="workdir/validation/memory_budget.csv")
args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}")

model = NanoTabPFNModel(embedding_size=192, num_attention_heads=6, mlp_hidden_size=768, num_layers=12,
                        num_outputs=10, feature_clip=10.0).to(device)
model.gradient_checkpointing = True
model.train()


def step(rows: int, columns: int) -> tuple[float, float] | None:
    """Peak memory (GB) and seconds of one training step on a rows x columns table; None if it does not fit."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    split = max(2, int(rows * args.train_share))
    try:
        x = torch.randn(1, rows, columns, device=device)
        y = torch.randint(0, 10, (1, rows), device=device).float()
        torch.cuda.synchronize(device)
        start = time.time()
        with autocast(device, torch.bfloat16):
            output = model((x, y[:, :split]), train_test_split_index=split)
        loss = torch.nn.functional.cross_entropy(output.float().reshape(-1, output.shape[-1]),
                                                 y[:, split:].reshape(-1).long())
        loss.backward()
        torch.cuda.synchronize(device)
        seconds = time.time() - start
        peak = torch.cuda.max_memory_allocated(device) / 1e9
    except torch.cuda.OutOfMemoryError:
        peak, seconds = float("inf"), float("nan")
    finally:
        model.zero_grad(set_to_none=True)
        x = y = output = loss = None  # noqa: F841 (free before the next try)
        torch.cuda.empty_cache()
    return None if peak > args.limit_gb else (peak, seconds)


rows_out = []
for columns in args.columns:
    low, high, best = 40, args.max_rows, None  # largest rows in [low, high] that fit, by bisection
    if step(low, columns) is None:
        print(f"{columns} columns: not even {low} rows fit", flush=True)
        continue
    while high - low > max(8, low // 50):
        middle = (low + high) // 2
        result = step(middle, columns)
        if result is None:
            high = middle
        else:
            low, best = middle, result
    if best is None:
        best = step(low, columns)
    peak, seconds = best
    rows_out.append({"columns": columns, "max_rows": low, "peak_gb": round(peak, 1),
                     "seconds_per_step": round(seconds, 2), "cells": low * columns})
    print(f"{columns:6d} columns: up to {low:6d} rows ({low * columns / 1e6:.2f} M cells), peak {peak:.1f} GB, "
          f"{seconds:.2f} s/step", flush=True)

out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows_out).to_csv(out, index=False)
print(pd.DataFrame(rows_out).to_string(index=False))
