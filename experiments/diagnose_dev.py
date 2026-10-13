"""Why a checkpoint ranks well (AUROC) but decides badly (accuracy) on development datasets: v3_p1 predicts the
minority class for nearly every patient of arcene and RELATHE. For each model, dataset, fold and variant it saves
what the model predicts, not only how well: training class frequencies, mean predicted probability and share of
predictions of each class, confidence, and the accuracy after rescaling the probabilities so that their mean
matches the training frequencies (if that fixes accuracy, the ranking is right and only the bias is wrong).

Variants, each changing one thing against `base` (what validate_run.py does: bf16, low memory, all numeric):
    base     as validate_run.py
    fp32     no bf16 autocast                                -> numerical precision?
    lowmem0  normal path instead of low memory               -> the low-memory path?
    clip100  feature clip 100 instead of the checkpoint's    -> the clipping of normalized values?
    log1p    sign(x) log(1 + |x|) on every value             -> heavy tails / sparse counts?
    cols500  500 random columns (same per fold)              -> width?
    rows300  300 training rows (stratified)                  -> context longer than in pretraining?
    swap     class indices reversed (y -> C-1-y)             -> a bias towards an index, or towards a frequency?

Same folds as validate_run.py. Resumable (rows already in the CSV are skipped). From the repo root:

    nohup python -u experiments/diagnose_dev.py --gpu 2 --models workdir/v2_p1_big/epoch_50.pth \\
        workdir/v3_p1/epoch_25.pth workdir/v3_p1/epoch_50.pth > workdir/validation/diagnose_dev.log 2>&1 &
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from tfmplayground.benchmarks import hdlss
from tfmplayground.benchmarks.metrics import accuracy, roc_auc
from tfmplayground.benchmarks.models import make_model
from tfmplayground.validation import BENCHMARK_DEV_DATASETS, DEV_FOLDER, dev_folds, load_dev_dataset

REPO = Path(__file__).resolve().parents[1]
VARIANTS = ("base", "fp32", "lowmem0", "clip100", "log1p", "cols500", "rows300", "swap")
parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", required=True, help="checkpoint paths")
parser.add_argument("--datasets", nargs="+", default=["arcene", "RELATHE", "orlraws10P", "lymphoma"])
parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--data-root", default=str(REPO.parent))
parser.add_argument("--output", default="workdir/validation/diagnose_dev.csv")
args = parser.parse_args()
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"


def load(dataset: str):
    if dataset in BENCHMARK_DEV_DATASETS:
        X, y, _ = hdlss.load_task(dataset, args.data_root)
        return X, y
    if not (Path(args.data_root) / DEV_FOLDER / f"{dataset}.mat").exists():
        raise FileNotFoundError(f"{dataset} not found under {args.data_root}")
    return load_dev_dataset(dataset, args.data_root)


def apply_variant(variant: str, fold: int, X_train, y_train, X_test):
    """Inputs of one variant; also returns how to map predicted columns back to the original classes."""
    rng = np.random.default_rng(fold)
    num_classes = int(y_train.max()) + 1
    back = np.arange(num_classes)
    if variant == "log1p":
        X_train, X_test = (np.sign(X) * np.log1p(np.abs(X)) for X in (X_train, X_test))
    elif variant == "cols500" and X_train.shape[1] > 500:
        columns = rng.choice(X_train.shape[1], 500, replace=False)
        X_train, X_test = X_train[:, columns], X_test[:, columns]
    elif variant == "rows300" and len(y_train) > 300:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=300, stratify=y_train,
                                                  random_state=fold)
    elif variant == "swap":
        y_train = num_classes - 1 - y_train
        back = back[::-1]  # predicted column k is original class C-1-k
    return X_train, y_train, X_test, back


def describe(y_train, y_test, proba) -> dict:
    """What the model predicts, besides how well."""
    num_classes = proba.shape[1]
    train_freq = np.bincount(y_train, minlength=num_classes) / len(y_train)
    mean_proba = proba.mean(axis=0)
    predicted = np.bincount(proba.argmax(axis=1), minlength=num_classes) / len(proba)
    rescaled = proba * (train_freq / np.clip(mean_proba, 1e-9, None))
    return {
        "roc_auc": roc_auc(y_test, proba),
        "accuracy": accuracy(y_test, proba),
        "balanced_accuracy": balanced_accuracy_score(y_test, proba.argmax(axis=1)),
        "log_loss": log_loss(y_test, np.clip(proba, 1e-7, 1), labels=np.arange(num_classes)),
        "accuracy_rescaled": accuracy(y_test, rescaled),
        "confidence": float(proba.max(axis=1).mean()),
        "majority_share": float(train_freq.max()),
        # largest gap between the share of predictions of a class and its training frequency
        "prediction_gap": float(np.abs(predicted - train_freq).max()),
        "train_freq": json.dumps(train_freq.round(3).tolist()),
        "mean_proba": json.dumps(mean_proba.round(3).tolist()),
        "predicted_share": json.dumps(predicted.round(3).tolist()),
    }


output = REPO / args.output
output.parent.mkdir(parents=True, exist_ok=True)
results = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=["model", "dataset", "fold", "variant"])
done = set(map(tuple, results[["model", "dataset", "fold", "variant"]].astype(str).values))

for dataset in args.datasets:
    X, y = load(dataset)
    for fold, (train, test) in enumerate(dev_folds(y)):
        for model_path in args.models:
            for variant in args.variants:
                if (model_path, dataset, str(fold), variant) in done:
                    continue
                X_train, y_train, X_test, back = apply_variant(variant, fold, X[train], y[train], X[test])
                model = make_model(
                    model_path, device, None if variant == "fp32" else torch.bfloat16, categorical=[],
                    low_memory=variant != "lowmem0", feature_clip=100.0 if variant == "clip100" else None,
                )
                model.fit(X_train, y_train)
                proba = np.asarray(model.predict_proba(X_test), dtype=np.float64)[:, back]
                row = {"model": model_path, "dataset": dataset, "fold": fold, "variant": variant,
                       **describe(y[train], y[test], proba)}  # fmt: skip
                results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                results.to_csv(output, index=False)
                print(f"{dataset} fold {fold} {variant:8s} {model_path}: AUROC {row['roc_auc']:.3f} "
                      f"acc {row['accuracy']:.3f} rescaled {row['accuracy_rescaled']:.3f} "
                      f"predicted {row['predicted_share']} train {row['train_freq']}", flush=True)  # fmt: skip

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
results["model"] = results["model"].str.replace("workdir/", "").str.replace(".pth", "")
for metric in ("accuracy", "accuracy_rescaled", "roc_auc", "prediction_gap", "confidence"):
    table = results.pivot_table(index=["dataset", "variant"], columns="model", values=metric, aggfunc="mean")
    print(f"\n{metric} (mean over folds)\n{table.round(3).to_string()}")
