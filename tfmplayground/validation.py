"""Fixed validation for research on priors and training settings, kept apart from the paper benchmarks, so a
training choice is never made on the data the paper reports.

Two parts:
- Held-out prior tables: tables drawn once from the prior with a fixed seed and saved, never trained on. The
  model's cross-entropy on them, at several table sizes, gives clean learning curves (the training loss mixes
  tables of every size and difficulty). Compared with the loss of predicting the training-class frequencies.
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
