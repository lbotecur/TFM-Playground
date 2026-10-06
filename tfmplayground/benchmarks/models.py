"""Models that can be evaluated, by name. Add a model by adding an entry to make_model().

Every model receives the same data: a float matrix whose categorical columns hold integer codes (NaN
for missing values) and the list of those columns. Each model is told about them in its own way.
"""

from __future__ import annotations

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASELINES = ("random_forest", "logreg", "logreg_en", "xgboost")
# Foundation models from other packages, by name. A suffix ":n<k>" sets the number of ensemble members
# (default 1, a single forward pass like our models and like TabPFN-Wide evaluates its own model);
# ":auto" uses the package default. TabPFN-3.5 sees at most 768 features per ensemble member (TabPFN v2,
# 500), so with one member it only sees a random subset of a wide table; "auto" adds members (8 to 32)
# until every feature is seen by at least one, up to 32 x 768 features.
# They need their own packages (tabpfnwide, which pins tabpfn 9.0.0, which includes TabPFN-3.5).
EXTERNAL = ("tabpfn-wide-5k", "tabpfn-v2-gn2p4bpt", "tabpfn-3.5")


def parse_external(name: str) -> tuple[str, int | str]:
    """'tabpfn-3.5:n8' -> ('tabpfn-3.5', 8); 'tabpfn-3.5:auto' -> ('tabpfn-3.5', 'auto');
    'tabpfn-3.5' -> ('tabpfn-3.5', 1)."""
    base, _, suffix = name.partition(":")
    if not suffix:
        return base, 1
    return base, "auto" if suffix == "auto" else int(suffix.removeprefix("n"))


def is_checkpoint(name: str) -> bool:
    """True for one of our checkpoints (a path), which is evaluated with NanoTabPFNClassifier."""
    return name not in BASELINES and parse_external(name)[0] not in EXTERNAL


def _elastic_net_cv():
    """Elastic-net logistic regression (l1_ratio 0.5), C chosen by 3-fold CV on the training rows: the
    usual linear baseline for omics. sklearn >= 1.8 infers the penalty from l1_ratios."""
    penalty = {} if tuple(map(int, sklearn.__version__.split(".")[:2])) >= (1, 8) else {"penalty": "elasticnet"}
    return LogisticRegressionCV(solver="saga", l1_ratios=[0.5], Cs=5, cv=3, max_iter=500, tol=1e-3,
                                scoring="neg_log_loss", n_jobs=-1, **penalty)


def _linear_inputs(categorical):
    """Linear models: one-hot for categorical columns; numeric columns imputed (median) and scaled."""
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    if not categorical:
        return numeric
    onehot = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
    return ColumnTransformer([("cat", onehot, list(categorical))], remainder=numeric)


def make_model(name: str, device: str = "cuda", amp_dtype=None, categorical: list[int] | None = None,
               low_memory: bool = False):
    """Baselines (BASELINES), other foundation models (EXTERNAL) or, for any other name, the path of one
    of our checkpoints, evaluated with NanoTabPFNClassifier without ensembling (one forward pass), as
    TabPFN-Wide evaluates its own model. amp_dtype (e.g. torch.bfloat16) only applies to our checkpoints.
    categorical: indices of the categorical columns ([] = all numeric). None keeps each model's own
    automatic detection (what the first MLOmics evaluations used). low_memory only applies to our
    checkpoints (see NanoTabPFNClassifier): for tables of tens of thousands of features."""
    if name == "random_forest":  # as TabPFN-Wide: default hyperparameters, NaN imputed with the mode
        return make_pipeline(SimpleImputer(strategy="most_frequent"), RandomForestClassifier(n_jobs=-1))
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="most_frequent"), StandardScaler(), LogisticRegression(max_iter=5000))
    if name == "logreg_en":
        return make_pipeline(_linear_inputs(categorical), _elastic_net_cv())
    if name == "xgboost":  # as TabPFN-Wide (analysis/baselines.py): 1000 trees, hist, on the GPU if given
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=1000, tree_method="hist", device=device, random_state=42, n_jobs=-1)
    base, n_estimators = parse_external(name)
    tabpfn_categorical = {} if categorical is None else {"categorical_features_indices": list(categorical)}
    if base == "tabpfn-wide-5k":  # as TabPFN-Wide evaluates it: one feature per group
        from tabpfnwide.classifier import TabPFNWideClassifier

        return TabPFNWideClassifier(model_name="wide-v2-5k", device=device, n_estimators=n_estimators,
                                    features_per_group=1, **tabpfn_categorical)
    if base == "tabpfn-v2-gn2p4bpt":  # TabPFN v2 checkpoint trained with features_per_group=1
        from huggingface_hub import hf_hub_download
        from tabpfn import TabPFNClassifier

        path = hf_hub_download(repo_id="Prior-Labs/TabPFN-v2-clf", filename="tabpfn-v2-classifier-gn2p4bpt.ckpt")
        return TabPFNClassifier(model_path=path, device=device, n_estimators=n_estimators, random_state=42,
                                ignore_pretraining_limits=True, **tabpfn_categorical)
    if base == "tabpfn-3.5":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5, device=device, n_estimators=n_estimators, random_state=42,
            ignore_pretraining_limits=True, **tabpfn_categorical,
        )
    from tfmplayground.interface import NanoTabPFNClassifier  # needs torch, only imported here

    if categorical is None:
        return NanoTabPFNClassifier(model=name, device=device, amp_dtype=amp_dtype, low_memory=low_memory)
    # Declared types only: without this, numeric columns with few distinct values (e.g. copy-number
    # levels, or genes with many zeros) would be inferred categorical.
    return NanoTabPFNClassifier(model=name, device=device, amp_dtype=amp_dtype, low_memory=low_memory,
                                categorical_features=list(categorical), infer_categorical=False)


def predict_proba(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model.fit(X_train, y_train)
    return np.asarray(model.predict_proba(X_test))
