"""Writes the release copy of a checkpoint: architecture (with the feature clip the model is meant to be used with)
and weights, without the optimizer state, so that NanoTabPFNClassifier(model=<copy>) gives the declared model with
no other setting.

The v2 of the paper, from the repo root:
    python experiments/export_checkpoint.py \\
        workdir/v2_p1_big-e100-rows2048-e100-w3000r1024-e30-lr3e-5/epoch_30.pth workdir/release/nanotabpfn_v2.pth \\
        --feature-clip 10
"""

import argparse
from pathlib import Path

from tfmplayground.interface import export_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("destination")
parser.add_argument("--feature-clip", type=float, default=None, help="default: the source's own (100 if it has none)")
args = parser.parse_args()

Path(args.destination).parent.mkdir(parents=True, exist_ok=True)
architecture = export_checkpoint(args.source, args.destination, args.feature_clip)
print(f"{args.destination}: {architecture}")
