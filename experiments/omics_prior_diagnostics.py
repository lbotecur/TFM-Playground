"""What tables does the mechanism omics prior (tfmplayground/priors/omics_mechanisms.py) make? Run before training on
it: the shape of the tables, and whether the label's mechanism carries a recoverable signal.

For every table, a random forest is cross-validated on the columns of the label's mechanism, the best a model can
use; gain = accuracy minus the majority class's share. With few rows that estimate is itself noisy (about +-0.07
with 40-80 rows), so the intrinsic signal is measured with --rows 1000: target < 10% of the tables with a signal
below a gain of 0.05. Every single-source copy of a mechanism variable is compared with its source by mutual
information with the label: a copy must carry less (target: it does in the large majority of pairs; with few rows
the estimate is noisy too).

    PYTHONPATH=. python experiments/omics_prior_diagnostics.py --tables 300               # tables as generated
    PYTHONPATH=. python experiments/omics_prior_diagnostics.py --tables 100 --rows 1000   # intrinsic signal
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score

from tfmplayground.priors.omics_mechanisms import COPY, LABEL_MECHANISM, MechanismPriorConfig, sample_dataset

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--tables", type=int, default=300)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--rows", type=int, default=None, help="rows of every table (default: as sampled)")
parser.add_argument("--output", default=None, help="CSV with one row per table")
args = parser.parse_args()
warnings.filterwarnings("ignore")


def gain(X: np.ndarray, y: np.ndarray) -> float:
    """3-fold accuracy of a random forest minus the majority share (missing values filled with the median)."""
    if X.shape[1] == 0:
        return np.nan
    X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    X = np.nan_to_num(X)
    folds = min(3, int(np.bincount(y).min()))
    if folds < 2:
        return np.nan
    forest = RandomForestClassifier(100, random_state=0, n_jobs=-1)
    return float(cross_val_score(forest, X, y, cv=folds).mean() - np.bincount(y).max() / len(y))


rng = np.random.default_rng(args.seed)
config = MechanismPriorConfig()
rows, start = [], time.time()
for i in range(args.tables):
    d = sample_dataset(rng, config, n_rows=args.rows)
    mech = d.role == LABEL_MECHANISM
    mech_cols = set(np.flatnonzero(mech).tolist())
    copies = np.array([d.role[j] == COPY and set(d.copy_sources[j]) <= mech_cols for j in range(len(d.role))])
    # pairs (copy, its single source) in the mechanism, the source informative (mutual information >= 0.02)
    pairs = [(j, d.copy_sources[j][0]) for j in np.flatnonzero(copies) if len(d.copy_sources[j]) == 1]
    copy_wins = np.nan
    if pairs and not d.is_null:
        cols = sorted({c for pair in pairs for c in pair})
        filled = np.nan_to_num(np.where(np.isnan(d.X[:, cols]), np.nanmedian(d.X[:, cols], axis=0), d.X[:, cols]))
        mi = dict(zip(cols, mutual_info_classif(filled, d.y, random_state=0), strict=True))
        informative = [(c, s) for c, s in pairs if mi[s] >= 0.02]  # a source with no information is a coin flip
        pairs = informative
        copy_wins = float(np.mean([mi[c] >= mi[s] for c, s in pairs])) if pairs else np.nan
    continuous = ~d.is_discrete
    Xc = d.X[:, continuous]
    rows.append({
        "rows": d.X.shape[0], "columns": d.X.shape[1], "classes": d.num_classes,
        "rows_per_class": d.X.shape[0] / d.num_classes, "layers": len(d.variant), "null": d.is_null,
        "label_mechanism": int(mech.sum()), "label_parents": int((d.distance == 1).sum()),
        "discrete_share": float(d.is_discrete.mean()), "missing_share": float(np.isnan(d.X).mean()),
        "zero_share_continuous": float(np.nanmean(Xc == 0)) if Xc.size else np.nan,
        "gain_mechanism": gain(d.X[:, mech], d.y), "gain_copies": gain(d.X[:, copies], d.y),
        "copy_pairs": len(pairs), "copy_beats_source": copy_wins,
        "layer_set": "+".join(sorted(d.variant)),
    })  # fmt: skip
    if (i + 1) % 50 == 0:
        print(f"{i + 1} tables, {time.time() - start:.0f}s", flush=True)

df = pd.DataFrame(rows)
if args.output:
    df.to_csv(args.output, index=False)
pd.set_option("display.width", 200)
numeric = df.drop(columns=["layer_set", "null"])
print("\n" + numeric.describe(percentiles=[0.1, 0.5, 0.9]).T[["mean", "10%", "50%", "90%"]].round(3).to_string())
signal = df[~df.null]
print(f"\nnull tables: {df.null.mean():.1%}")
print(f"tables with a signal whose mechanism gains < 0.05: {(signal.gain_mechanism < 0.05).mean():.1%} (target < 10%)")
paired = signal.dropna(subset=["copy_beats_source"])
share = np.average(paired.copy_beats_source, weights=paired.copy_pairs) if len(paired) else np.nan
print(f"(copy, source) pairs where the copy carries at least as much information on the label: {share:.1%} "
      f"of {int(paired.copy_pairs.sum())} pairs")  # fmt: skip
print(f"single-omic tables: {(df.layers == 1).mean():.1%}; most frequent layer sets:")
print(df.layer_set.value_counts().head(8).to_string())
