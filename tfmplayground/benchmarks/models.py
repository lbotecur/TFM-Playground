"""Models that can be evaluated, by name. Add a model by adding an entry to make_model()."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Foundation models from other packages, by name. A suffix ":n<k>" sets the number of ensemble members
# (default 1, a single forward pass like our models and like TabPFN-Wide evaluates its own model).
# They need their own packages (tabpfnwide, which pins tabpfn 9.0.0, which includes TabPFN-3.5).
EXTERNAL = ("tabpfn-wide-5k", "tabpfn-v2-gn2p4bpt", "tabpfn-3.5")


def parse_external(name: str) -> tuple[str, int]:
    """'tabpfn-3.5:n8' -> ('tabpfn-3.5', 8); 'tabpfn-3.5' -> ('tabpfn-3.5', 1)."""
    base, _, n = name.partition(":n")
    return base, int(n) if n else 1


def is_checkpoint(name: str) -> bool:
    """True for one of our checkpoints (a path), which is evaluated with NanoTabPFNClassifier."""
    return name not in ("random_forest", "logreg") and parse_external(name)[0] not in EXTERNAL


def make_model(name: str, device: str = "cuda", amp_dtype=None):
    """'random_forest' and 'logreg' are baselines; names in EXTERNAL are other foundation models; any
    other name is taken as the path of one of our checkpoints and evaluated with NanoTabPFNClassifier,
    without ensembling (one forward pass), as TabPFN-Wide evaluates its own model. amp_dtype (e.g.
    torch.bfloat16) only applies to our checkpoints."""
    if name == "random_forest":  # as TabPFN-Wide: default hyperparameters, NaN imputed with the mode
        return make_pipeline(SimpleImputer(strategy="most_frequent"), RandomForestClassifier(n_jobs=-1))
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="most_frequent"), StandardScaler(), LogisticRegression(max_iter=5000))
    base, n_estimators = parse_external(name)
    if base == "tabpfn-wide-5k":  # as TabPFN-Wide evaluates it: one feature per group
        from tabpfnwide.classifier import TabPFNWideClassifier

        return TabPFNWideClassifier(model_name="wide-v2-5k", device=device, n_estimators=n_estimators,
                                    features_per_group=1)
    if base == "tabpfn-v2-gn2p4bpt":  # TabPFN v2 checkpoint trained with features_per_group=1
        from huggingface_hub import hf_hub_download
        from tabpfn import TabPFNClassifier

        path = hf_hub_download(repo_id="Prior-Labs/TabPFN-v2-clf", filename="tabpfn-v2-classifier-gn2p4bpt.ckpt")
        return TabPFNClassifier(model_path=path, device=device, n_estimators=n_estimators, random_state=42,
                                ignore_pretraining_limits=True)
    if base == "tabpfn-3.5":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5, device=device, n_estimators=n_estimators, random_state=42,
            ignore_pretraining_limits=True,
        )
    from tfmplayground.interface import NanoTabPFNClassifier  # needs torch, only imported here

    return NanoTabPFNClassifier(model=name, device=device, amp_dtype=amp_dtype)


def predict_proba(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model.fit(X_train, y_train)
    return np.asarray(model.predict_proba(X_test))
