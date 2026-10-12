"""Feature importance of our checkpoints, TabPFN-Wide and a random forest, scored the same way, for the
interpretability comparison of the paper.

Our models and TabPFN-Wide give the attention of the label (target) token to each feature, averaged over rows,
heads and layers (TabPFN-Wide, Section 3.4 of the paper; NanoTabPFNClassifier.feature_attention_scores). The
random forest gives impurity-based importance, the reference TabPFN-Wide used on synthetic data. Every model is
fitted on the same training rows and scored on the same test rows, so the importance can be read together with
how well the model predicts.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

WIDE = "tabpfn-wide-5k"
RANDOM_FOREST = "random_forest"
ANOVA = "anova"  # univariate F-test, the classic filter for omics feature selection

# The 50 genes of the PAM50 breast cancer subtype classifier (Parker et al., 2009), which define the BRCA labels
# of MLOmics, with the older names some expression tables still use.
PAM50 = ("ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20", "CDC6", "CDH3", "CENPF",
         "CEP55", "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C",
         "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK", "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1",
         "NDC80", "NUF2", "ORC6", "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS", "UBE2C",
         "UBE2T")
PAM50_ALIASES = {"ORC6L": "ORC6", "KNTC2": "NDC80", "CDCA1": "NUF2"}


def fit_and_explain(name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, device: str = "cuda",
                    amp_dtype: torch.dtype | None = torch.bfloat16, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fits a model on (X_train, y_train) and returns (scores, proba): one importance score per column
    (higher = more important; NaN for columns a model dropped) and the predicted class probabilities of
    X_test, so that what a model says it uses can be judged together with how well it predicts.
    The attention scores come from the forward pass that predicts X_test (training and test rows).
    name: "random_forest", "tabpfn-wide-5k", "anova" (F statistic, proba None) or the path of one of our
    checkpoints."""
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    if name == ANOVA:  # no predictions: a filter, not a model
        from sklearn.feature_selection import f_classif

        with np.errstate(divide="ignore", invalid="ignore"):
            f = f_classif(X_train, y_train)[0]
        return np.where(np.isfinite(f), f, np.nan), None
    if name == RANDOM_FOREST:
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1).fit(X_train, y_train)
        return rf.feature_importances_, rf.predict_proba(X_test)
    if name == WIDE:  # needs the tabpfnwide package (tfm_ext)
        from tabpfnwide.classifier import TabPFNWideClassifier

        # one estimator and one feature per group: one token per feature (required for attention);
        # MIN_UNIQUE_FOR_NUMERICAL_FEATURES=1 keeps low-cardinality numeric columns numeric (its README)
        clf = TabPFNWideClassifier(model_name="wide-v2-5k", device=device, n_estimators=1, features_per_group=1,
                                   save_attention_maps="label", random_state=seed,
                                   inference_config={"MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 1})
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        return np.asarray(clf.get_attention_to_label(), dtype=float), np.asarray(proba)
    from tfmplayground.interface import NanoTabPFNClassifier

    clf = NanoTabPFNClassifier(model=name, device=device, amp_dtype=amp_dtype, categorical_features=[],
                               infer_categorical=False).fit(X_train, y_train)
    return np.asarray(clf.feature_attention_scores(X_test), dtype=float), clf.predict_proba(X_test)


def feature_importance(name: str, X: np.ndarray, y: np.ndarray, device: str = "cuda",
                       amp_dtype: torch.dtype | None = torch.bfloat16, seed: int = 0) -> np.ndarray:
    """Importance scores of a model fitted on (X, y), with X as the query rows (see fit_and_explain)."""
    return fit_and_explain(name, X, y, X, device=device, amp_dtype=amp_dtype, seed=seed)[0]


def _ranked(scores: np.ndarray) -> np.ndarray:
    """NaN (a column the model dropped) ranks below every scored column."""
    scores = np.asarray(scores, dtype=float)
    lowest = np.nanmin(scores) - 1.0 if np.isfinite(scores).any() else 0.0
    return np.where(np.isnan(scores), lowest, scores)


def recall_at_k(scores: np.ndarray, informative: np.ndarray) -> float:
    """Share of the k informative columns among the k top-scored ones (TabPFN-Wide's Recall@k)."""
    informative = np.asarray(informative)
    k = len(informative)
    top = np.argsort(-_ranked(scores), kind="stable")[:k]
    return len(set(top.tolist()) & set(informative.tolist())) / k


def informative_auroc(scores: np.ndarray, informative: np.ndarray) -> float:
    """Probability that an informative column outscores a noise column (1 = perfect separation, 0.5 = chance):
    uses the whole ranking, not only the top k."""
    is_informative = np.zeros(len(scores), dtype=bool)
    is_informative[np.asarray(informative)] = True
    return float(roc_auc_score(is_informative, _ranked(scores)))


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores (NaN last)."""
    return np.argsort(-_ranked(scores), kind="stable")[:k]


def jaccard(a, b) -> float:
    a, b = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    return len(a & b) / len(a | b) if a | b else float("nan")


def gene_symbol(column: str) -> str:
    """'ESR1', 'ESR1|2099' or 'ESR1.1' -> 'ESR1', with old PAM50 names mapped to the current ones."""
    symbol = str(column).split("|")[0].strip().upper()
    return PAM50_ALIASES.get(symbol, symbol)


def hits_and_enrichment(ranked_symbols: list[str], gene_set: set[str], universe: set[str], k: int) -> tuple[int, float]:
    """How many of the k top-ranked genes are in gene_set, and the hypergeometric p-value of at least that many
    by chance (gene_set restricted to the genes present)."""
    from scipy.stats import hypergeom

    present = gene_set & universe
    hits = len(set(ranked_symbols[:k]) & present)
    return hits, float(hypergeom.sf(hits - 1, len(universe), len(present), k))
