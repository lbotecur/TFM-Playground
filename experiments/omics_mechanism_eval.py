"""Do the models predict new tables of the mechanism omics prior, and does their feature importance find the label's
mechanism rather than its copies? A first check of a model trained with --omics-fraction against its control (the
definitive version for the interpretability paper lives in tabomics).

Tables come from tfmplayground/priors/omics_mechanisms.py with a seed never used in training; each is split 70/30
(stratified). Per table and model:
- prediction: AUROC (one-vs-rest) and accuracy on the test rows;
- importance (tables whose label depends on something): AUROC of the scores between the label's mechanism and every
  other column; the same against only the copies of the mechanism; Recall@k of the mechanism (k = its size); AUROC of
  the label's parents against the rest; and the share of (variable, copy) pairs where the variable scores higher.
Our checkpoints give the attention of the label to each column (NanoTabPFNClassifier.feature_attention_scores, all
rows, as in experiments/interpretability_synthetic.py); random_forest and anova are references.

    python experiments/omics_mechanism_eval.py --gpu 0 --tables 200 --output workdir/eval/omics_mechanism.csv \\
        --models workdir/v2_p1_big-e300-w3000r1024-e30-lr3e-5/epoch_30.pth workdir/v2_e300-final-control/epoch_30.pth \\
                 workdir/v2_e300-final-omics30/epoch_30.pth random_forest anova
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tfmplayground.interpretability import ANOVA, RANDOM_FOREST, fit_and_explain
from tfmplayground.priors.omics_mechanisms import COPY, LABEL_MECHANISM, MechanismPriorConfig, sample_dataset

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--models", nargs="+", required=True, help="checkpoint paths, random_forest, anova")
parser.add_argument("--tables", type=int, default=200)
parser.add_argument("--seed", type=int, default=20261016, help="never used in training")
parser.add_argument("--max-features", type=int, default=3000)
parser.add_argument("--max-rows", type=int, default=1024)
parser.add_argument("--feature-clip", type=float, default=10.0, help="our checkpoints, as the v2 is used")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--output", default=None, help="CSV with one row per table and model")
args = parser.parse_args()
warnings.filterwarnings("ignore")
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"


def explain(name: str, X_train, y_train, X_test):
    """(scores, proba) of one model; our checkpoints in bf16 with the given feature clip."""
    if name in (RANDOM_FOREST, ANOVA):  # no missing values there: the training rows' median fills them
        median = np.nan_to_num(np.nanmedian(X_train, axis=0))
        X_train, X_test = (np.where(np.isnan(x), median, x) for x in (X_train, X_test))
        return fit_and_explain(name, X_train, y_train, X_test, device=device)
    from tfmplayground.interface import NanoTabPFNClassifier

    amp = torch.bfloat16 if device.startswith("cuda") else None
    clf = NanoTabPFNClassifier(model=name, device=device, amp_dtype=amp, categorical_features=[],
                               infer_categorical=False, feature_clip=args.feature_clip)  # fmt: skip
    clf.fit(X_train, y_train)
    return np.asarray(clf.feature_attention_scores(X_test), dtype=float), clf.predict_proba(X_test)


def auroc(positive: np.ndarray, scores: np.ndarray, among: np.ndarray) -> float:
    """AUROC of the scores between `positive` and the other columns of `among` (both boolean masks)."""
    y = positive[among]
    return float(roc_auc_score(y, scores[among])) if 0 < y.sum() < len(y) else np.nan


def importance_metrics(d, scores: np.ndarray) -> dict:
    scores = np.where(np.isnan(scores), -np.inf, scores)  # dropped (constant) columns rank last
    scores = np.where(np.isinf(scores), np.nanmin(scores[np.isfinite(scores)]) - 1, scores)
    mech = d.role == LABEL_MECHANISM
    mech_cols = set(np.flatnonzero(mech).tolist())
    copies = np.array([d.role[j] == COPY and set(d.copy_sources[j]) <= mech_cols for j in range(len(d.role))])
    k = int(mech.sum())
    pairs = [(d.copy_sources[j][0], j) for j in np.flatnonzero(copies) if len(d.copy_sources[j]) == 1]
    return {
        "imp_auroc_mechanism": auroc(mech, scores, np.ones_like(mech)),
        "imp_auroc_vs_copies": auroc(mech, scores, mech | copies),
        "imp_recall_at_k": float(mech[np.argsort(-scores)[:k]].mean()),
        "imp_auroc_parents": auroc(d.distance == 1, scores, np.ones_like(mech)),
        "original_beats_copy": float(np.mean([scores[s] > scores[c] for s, c in pairs])) if pairs else np.nan,
    }


rng = np.random.default_rng(args.seed)
config = MechanismPriorConfig(max_rows=args.max_rows, max_total_features=args.max_features)
rows, start = [], time.time()
for i in range(args.tables):
    d = sample_dataset(rng, config)
    stratify = d.y if np.bincount(d.y).min() >= 2 else None
    tr, te = train_test_split(np.arange(len(d.y)), test_size=0.3, stratify=stratify, random_state=i)
    if len(np.unique(d.y[tr])) < d.num_classes:  # the model must see every class in context
        continue
    labels = np.arange(d.num_classes)
    for name in args.models:
        scores, proba = explain(name, d.X[tr], d.y[tr], d.X[te])
        row = {"table": i, "model": name, "rows": len(d.y), "columns": d.X.shape[1], "classes": d.num_classes,
               "null": d.is_null, "mechanism_size": int((d.role == LABEL_MECHANISM).sum())}  # fmt: skip
        if proba is not None:
            proba = np.asarray(proba)
            y_te = d.y[te]
            try:
                row["auroc"] = float(roc_auc_score(y_te, proba[:, 1]) if d.num_classes == 2 else
                                     roc_auc_score(y_te, proba, multi_class="ovr", labels=labels))  # fmt: skip
            except ValueError:  # a class missing from the test rows
                row["auroc"] = np.nan
            row["accuracy"] = float(accuracy_score(y_te, proba.argmax(axis=1)))
        if not d.is_null:
            row.update(importance_metrics(d, scores))
        rows.append(row)
    if (i + 1) % 20 == 0:
        print(f"{i + 1} tables, {time.time() - start:.0f}s", flush=True)

df = pd.DataFrame(rows)
if args.output:
    df.to_csv(args.output, index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 70)
metrics = ["auroc", "accuracy", "imp_auroc_mechanism", "imp_auroc_vs_copies", "imp_recall_at_k", "imp_auroc_parents",
           "original_beats_copy"]  # fmt: skip
metrics = [m for m in metrics if m in df.columns]
print(f"\nMeans over {df.table.nunique()} tables ({df[~df.null].table.nunique()} with a signal for the importance)\n")
print(df.groupby("model", sort=False)[metrics].mean().round(4).to_string())
reference = args.models[0]
print(f"\nPer table against {reference}: wins / losses (ties left out)\n")
wide = df.pivot_table(index="table", columns="model", values=metrics)
for name in args.models[1:]:
    parts = []
    for m in metrics:
        if (m, name) in wide.columns and (m, reference) in wide.columns:
            diff = (wide[(m, name)] - wide[(m, reference)]).dropna()
            parts.append(f"{m} {int((diff > 0).sum())}/{int((diff < 0).sum())}")
    print(f"{name}: " + ", ".join(parts))
