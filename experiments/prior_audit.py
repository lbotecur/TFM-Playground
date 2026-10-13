"""What graph_scm tables look like before and after TabICL's preprocessing, and what reaches the model.

Generates graph_scm_raw tables (TabICL's graph_scm without its clamp and z-score, with categorical masks) and,
for each, TabICL's own version (raw + clamp to ~4 robust std + z-score over all rows, bit for bit what
graph_scm gives; see tests/test_graph_raw_prior.py). Prints:
  1. columns: share of categorical ones, their cardinality and rarest category;
  2. the clamp: numeric columns and values it changes, categorical columns where it merges a category;
  3. the raw scale (largest |x| and column std per table, infinite or missing values);
  4. what the model sees: |z| after normalize_features (train rows, no clip) for both versions, and columns
     constant on the train rows only.
CPU only, about 1 s per 8 tables. From the repo root:
    python experiments/prior_audit.py --tables 1280 --seed 0
Results of 2026-10-12/13 in docs/prior_audit.md.
"""

import argparse
import random

import numpy as np
import pandas as pd
import torch
from tabicl.prior._reg2cls import outlier_removing, standard_scaling

from tfmplayground.external_priors.graph_raw import RawGraphPrior
from tfmplayground.models.nanotabpfn import normalize_features

parser = argparse.ArgumentParser()
parser.add_argument("--tables", type=int, default=1280)
parser.add_argument("--min-rows", type=int, default=40)
parser.add_argument("--max-rows", type=int, default=1024)
parser.add_argument("--min-features", type=int, default=2)
parser.add_argument("--max-features", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
prior = RawGraphPrior(batch_size=8, batch_size_per_gp=8, min_features=args.min_features,
                      max_features=args.max_features, max_classes=10, min_seq_len=args.min_rows,
                      max_seq_len=args.max_rows, log_seq_len=True, min_train_size=0.3, max_train_size=0.9,
                      n_jobs=1, raw=True)

columns, tables = [], []
z_raw, z_tabicl = [], []
while len(tables) < args.tables:
    X, y, d, seq_lens, train_sizes, masks = prior.get_batch()
    for i in range(X.shape[0]):
        raw = X[i, : seq_lens[i], : d[i]].float()
        clamped = outlier_removing(raw, threshold=4)
        tabicl = standard_scaling(clamped)
        split = int(train_sizes[i])
        finite = raw[~torch.isnan(raw)]
        tables.append(dict(rows=raw.shape[0], features=raw.shape[1], max_abs=float(finite.abs().max()),
                           max_std=float(torch.nan_to_num(raw.std(0)).max()), inf=bool(torch.isinf(finite).any()),
                           nan=bool(torch.isnan(raw).any()),
                           constant_on_train=int((raw[:split].std(0) < 1e-6).sum())))
        for j in range(raw.shape[1]):
            a, b = raw[:, j], clamped[:, j]
            values, counts = torch.unique(a, return_counts=True)
            columns.append(dict(categorical=bool(masks[i, j]), k=len(values), rarest=float(counts.min()) / len(a),
                                changed=float((a != b).float().mean()), k_after=len(torch.unique(b))))
        for source, store in ((raw, z_raw), (tabicl, z_tabicl)):
            store.append(normalize_features(source[None], split, clip=1e9)[0, :, :, 0].abs().flatten())

cols, tabs = pd.DataFrame(columns), pd.DataFrame(tables)
cat, num = cols[cols.categorical], cols[~cols.categorical]
pd.set_option("display.width", 200)
print(f"{len(tabs)} tables, {len(cols)} columns ({args.min_rows}-{args.max_rows} rows, "
      f"{args.min_features}-{args.max_features} features, seed {args.seed})\n")
print("1. Columns")
print(f"  categorical: {len(cat) / len(cols):.1%} of the columns")
print(f"  categories per categorical column, quantiles 50/90/99/max: "
      f"{cat.k.quantile(.5):.0f} / {cat.k.quantile(.9):.0f} / {cat.k.quantile(.99):.0f} / {cat.k.max()}")
print(f"  rarest category below 1/17 of the rows (beyond 4 std when binary): {(cat.rarest < 1 / 17).mean():.1%}; "
      f"below 1%: {(cat.rarest < .01).mean():.1%}")
print("2. TabICL's clamp (4 robust std, all rows)")
print(f"  numeric columns with a value changed: {(num.changed > 0).mean():.1%}; values changed: {num.changed.mean():.2%}")
print(f"  categorical columns that lose a category: {(cat.k_after < cat.k).mean():.1%}; "
      f"columns made constant: {((cols.k_after <= 1) & (cols.k > 1)).mean():.2%}")
print("3. Raw scale per table, quantiles 50/90/99/max")
for name in ("max_abs", "max_std"):
    q = tabs[name].quantile([.5, .9, .99, 1]).values
    print(f"  {name}: " + " / ".join(f"{v:.3g}" for v in q))
print(f"  tables with infinite values: {tabs.inf.sum()}, with missing values: {tabs.nan.sum()}")
print("4. |z| reaching the model (normalize_features with the train rows, before the clip): share of values above")
for name, zs in (("raw (graph_scm_raw)", z_raw), ("TabICL (graph_scm)", z_tabicl)):
    z = torch.cat(zs)
    print(f"  {name:20s} " + "  ".join(f">{t}: {(z > t).float().mean():.2e}" for t in (4, 6, 10, 100)))
print(f"  columns constant on the train rows only: {tabs.constant_on_train.sum() / len(cols):.2%} "
      "(z is huge on the test rows; the clip bounds it)")
