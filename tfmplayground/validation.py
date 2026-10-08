"""Fixed validation for research on priors and training settings, kept apart from the paper benchmarks, so a
training choice is never made on the data the paper reports.

Two parts:
- Held-out prior tables: tables drawn once from the prior with a fixed seed and saved, never trained on. The
  model's cross-entropy on them, at several table sizes, gives clean learning curves (the training loss mixes
  tables of every size and difficulty). Compared with the loss of predicting the training-class frequencies.
- Development datasets: real high-dimensional datasets from scikit-feature that are not among the 15 of the
  HDLSS benchmark (nor in any other paper benchmark), in their own folder, to check that what improves on
  the prior also improves on real data.
- Synthetic probes: tasks with a known structure (XOR of 2 and 3 features, a linear signal among 480 noise
  features, madelon's own generator) at several training sizes. They measure whether the model learned to use
  feature interactions and to ignore noise, which the held-out loss averages away.
"""

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

# Held-out prior tables: rows per table, feature range, extra (widening) features and the prior.
PRIOR_SETS = {
    "graph_r100": dict(prior_type="graph_scm", rows=100, min_features=2, max_features=100, widen=0),
    "graph_r300": dict(prior_type="graph_scm", rows=300, min_features=2, max_features=100, widen=0),
    "graph_r1000": dict(prior_type="graph_scm", rows=1000, min_features=2, max_features=100, widen=0),
    "graph_r300_wide2000": dict(prior_type="graph_scm", rows=300, min_features=2, max_features=100, widen=2000),
    "tree_r300": dict(prior_type="tree_scm", rows=300, min_features=2, max_features=100, widen=0),
}
TRAIN_FRACTION = 0.7  # context = 70% of the rows: 70, 210 and 700 rows

# Synthetic probes: (task, training rows, pure-noise features added).
PROBES = [
    ("xor2", 100, 0), ("xor2", 300, 0), ("xor2", 1000, 0),
    ("xor3", 100, 0), ("xor3", 300, 0), ("xor3", 1000, 0), ("xor3", 1000, 20),
    ("linear", 300, 480),
    ("madelon_like", 1000, 0), ("madelon_like", 1000, 480),
]
PROBE_TEST_ROWS = 500

# Development datasets (scikit-feature .mat files in <data_root>/HDLSS_dev): gene expression (GLIOMA, lymphoma,
# nci9, lung_discrete), images as wide tables (warpPIE10P, orlraws10P) and many rows (USPS). None of them is in
# HDLSS, MLOmics, Shamir or TabArena. Carcinom is left out: 11 classes, more than the model's 10.
DEV_FOLDER = "HDLSS_dev"
DEV_DATASETS = ("GLIOMA", "lymphoma", "nci9", "lung_discrete", "warpPIE10P", "orlraws10P", "USPS")
DEV_SPLITS, DEV_SEED, DEV_LARGE = 3, 42, 2500  # stratified 3-fold, 2 repeats (1 with DEV_LARGE rows or more)


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_prior_set(name: str, n_tables: int, seed: int = 0) -> list[dict]:
    """n_tables tables of PRIOR_SETS[name], one at a time, each as {'x', 'y', 'split'} on the CPU."""
    from tfmplayground.augmentation import add_mixed_widening_features
    from tfmplayground.external_priors.tabicl import TabICLPriorDataLoader

    spec = PRIOR_SETS[name]
    _seed_everything(seed)
    loader = TabICLPriorDataLoader(
        num_steps=n_tables, batch_size=1, num_datapoints_min=spec["rows"],
        num_datapoints_max=spec["rows"] + 1,  # TabICL draws rows in [min, max): exactly spec["rows"]
        min_features=spec["min_features"], max_features=spec["max_features"], max_num_classes=10,
        device=torch.device("cpu"), prior_type=spec["prior_type"], log_seq_len=False,
        min_train_size=TRAIN_FRACTION, max_train_size=TRAIN_FRACTION,
    )
    generator = torch.Generator().manual_seed(seed)
    tables = []
    for batch in loader:
        x = batch["x"]
        if spec["widen"]:  # as training widens: sparsity U(0, 0.05), noise U(0, 1)
            sparsity = torch.rand(1, generator=generator).item() * 0.05
            noise = torch.rand(1, generator=generator).item()
            x = add_mixed_widening_features(x, spec["widen"], sparsity, noise, generator=generator)
        tables.append({"x": x[0], "y": batch["y"][0], "split": int(batch["train_test_split_index"])})
    return tables


def load_prior_sets(directory: str | Path, n_tables: int = 200, names=None) -> dict[str, list[dict]]:
    """The held-out tables, built and saved in directory the first time (every run then uses the same file)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sets = {}
    for i, name in enumerate(names or PRIOR_SETS):
        path = directory / f"{name}_n{n_tables}.pt"
        if not path.exists():
            torch.save(build_prior_set(name, n_tables, seed=1000 + i), path)
        sets[name] = torch.load(path, weights_only=False)
    return sets


@torch.no_grad()
def prior_metrics(model, tables: list[dict], device, amp_dtype=torch.bfloat16) -> dict:
    """Mean over tables of the test rows' cross-entropy (as in training: logits over all outputs) and accuracy,
    plus the cross-entropy of predicting the training-class frequencies (the loss with nothing learned)."""
    from tfmplayground.utils import autocast

    model.eval()
    loss, accuracy, baseline = [], [], []
    for t in tables:
        x, y, split = t["x"][None].to(device), t["y"].to(device), t["split"]
        with autocast(device, amp_dtype):
            logits = model((x, y[None, :split]), train_test_split_index=split)
        logits = logits[0].float()
        target = y[split:].long()
        loss.append(torch.nn.functional.cross_entropy(logits, target).item())
        accuracy.append((logits.argmax(-1) == target).float().mean().item())
        counts = torch.bincount(y[:split].long(), minlength=logits.shape[-1]).float() + 0.5  # smoothed
        baseline.append(-torch.log(counts / counts.sum())[target].mean().item())
    return {"loss": float(np.mean(loss)), "accuracy": float(np.mean(accuracy)), "baseline_loss": float(np.mean(baseline))}


def prior_metrics_classifier(name: str, tables: list[dict], device) -> dict:
    """Held-out prior loss of any model through its classifier interface (fit on the training rows, predict the
    test rows), so ours and other models are compared on the same tables. Only test rows whose class appears in
    the training rows count, since no classifier can give probability to a class it never saw. The reference
    is TabICL v2, trained at length on this same prior: its loss estimates how low the prior lets a model go,
    and the gap to ours how much training can still gain."""
    from tfmplayground.benchmarks.models import is_checkpoint, make_model, predict_proba

    ours = is_checkpoint(name)
    loss, accuracy = [], []
    for t in tables:
        x, y, split = t["x"].numpy(), t["y"].long().numpy(), t["split"]
        y_train, y_test = y[:split], y[split:]
        seen = np.isin(y_test, y_train)
        if len(np.unique(y_train)) < 2 or not seen.any():
            continue
        model = make_model(name, device, torch.bfloat16 if ours else None, categorical=[])
        proba = predict_proba(model, x[:split], y_train, x[split:][seen])
        classes = np.unique(y_train)  # the column order of predict_proba for every model here
        p_true = proba[np.arange(seen.sum()), np.searchsorted(classes, y_test[seen])]
        loss.append(float(-np.log(np.clip(p_true, 1e-7, 1)).mean()))
        accuracy.append(float((classes[proba.argmax(1)] == y_test[seen]).mean()))
    return {"loss": float(np.mean(loss)), "accuracy": float(np.mean(accuracy))}


def load_dev_dataset(name: str, data_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Features and labels 0..C-1 of a development dataset, rows shuffled with a fixed seed, classes with fewer
    rows than folds dropped (so every class is in every training fold)."""
    import scipy.sparse
    from scipy.io import loadmat
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils import shuffle

    data = loadmat(Path(data_root) / DEV_FOLDER / f"{name}.mat")
    X, y = data["X"], np.asarray(data["Y"]).ravel()
    if scipy.sparse.issparse(X):
        X = X.toarray()
    X, y = shuffle(np.asarray(X, dtype=np.float32), y, random_state=DEV_SEED)
    labels, counts = np.unique(y, return_counts=True)
    keep = np.isin(y, labels[counts >= DEV_SPLITS])
    return X[keep], LabelEncoder().fit_transform(y[keep])


def dev_folds(y: np.ndarray):
    from sklearn.model_selection import RepeatedStratifiedKFold

    repeats = 2 if len(y) < DEV_LARGE else 1
    cv = RepeatedStratifiedKFold(n_splits=DEV_SPLITS, n_repeats=repeats, random_state=DEV_SEED)
    return list(cv.split(np.zeros(len(y)), y))


def run_dev(name: str, device: str, data_root: str | Path, datasets=DEV_DATASETS) -> list[dict]:
    """AUROC and accuracy of model `name` on every fold of the development datasets (set 'dev_<dataset>', seed =
    fold). Datasets whose file is missing are skipped."""
    from tfmplayground.benchmarks.metrics import accuracy, roc_auc
    from tfmplayground.benchmarks.models import is_checkpoint, make_model, predict_proba

    ours = is_checkpoint(name)
    rows = []
    for dataset in datasets:
        if not (Path(data_root) / DEV_FOLDER / f"{dataset}.mat").exists():
            continue
        X, y = load_dev_dataset(dataset, data_root)
        for fold, (train, test) in enumerate(dev_folds(y)):
            model = make_model(name, device, torch.bfloat16 if ours else None, categorical=[])
            proba = predict_proba(model, X[train], y[train], X[test])
            rows += [{"set": f"dev_{dataset}", "seed": fold, "metric": "roc_auc", "value": roc_auc(y[test], proba)},
                     {"set": f"dev_{dataset}", "seed": fold, "metric": "accuracy", "value": accuracy(y[test], proba)}]
    return rows


def probe_signal(task: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Relevant features and binary labels of one probe task."""
    if task == "linear":  # 5 features, labels from a noisy linear score: no interaction needed
        X = rng.normal(size=(n, 5))
        return X, (X @ rng.normal(size=5) + 0.5 * rng.normal(size=n) > 0).astype(int)
    if task in ("xor2", "xor3"):  # the class is the sign of a product: each feature alone says nothing
        k = int(task[-1])
        X = rng.normal(size=(n, k))
        return X, (np.prod(X, axis=1) > 0).astype(int)
    if task == "madelon_like":  # madelon's own generator (sklearn's make_classification comes from it): 5
        # informative features on the vertices of a hypercube, 16 clusters per class, 15 linear combinations
        return make_classification(n_samples=n, n_features=20, n_informative=5, n_redundant=15, n_repeated=0,
                                   n_clusters_per_class=16, flip_y=0.01, class_sep=1.0, shuffle=False,
                                   random_state=int(rng.integers(1 << 31)))
    raise ValueError(task)


def probe_data(task: str, n_train: int, noise: int, seed: int):
    """X_train, y_train, X_test, y_test of one probe: the task's features plus `noise` pure-noise features, all
    at random positions, PROBE_TEST_ROWS test rows."""
    rng = np.random.default_rng(seed)
    X, y = probe_signal(task, n_train + PROBE_TEST_ROWS, rng)
    X = np.hstack([X, rng.normal(size=(len(X), noise))]).astype(np.float32)
    X = X[:, rng.permutation(X.shape[1])]
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def probe_name(task: str, n_train: int, noise: int) -> str:
    return f"{task}_n{n_train}_noise{noise}"


def run_probes(name: str, device: str, seeds: int = 5, probes=PROBES) -> list[dict]:
    """AUROC of model `name` (a checkpoint path or any make_model name) on every probe and seed."""
    from tfmplayground.benchmarks.models import is_checkpoint, make_model, predict_proba

    ours = is_checkpoint(name)
    rows = []
    for task, n_train, noise in probes:
        for seed in range(seeds):
            X_train, y_train, X_test, y_test = probe_data(task, n_train, noise, seed)
            model = make_model(name, device, torch.bfloat16 if ours else None, categorical=[])
            proba = predict_proba(model, X_train, y_train, X_test)
            rows.append({"set": probe_name(task, n_train, noise), "seed": seed, "metric": "roc_auc",
                         "value": roc_auc_score(y_test, proba[:, 1])})
    return rows
