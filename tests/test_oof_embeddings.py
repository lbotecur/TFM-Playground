import numpy as np
import pandas as pd
import pytest
import torch

from tfmplayground.embedding import leave_one_fold_out_embeddings
from tfmplayground.interface import NanoTabPFNClassifier, NanoTabPFNRegressor
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def _clf():
    torch.manual_seed(0)
    model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=3)
    return NanoTabPFNClassifier(model=model, device="cpu")


def _reg():
    torch.manual_seed(0)
    model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=8)
    return NanoTabPFNRegressor(model=model, device="cpu")


def test_leave_one_fold_out_embeddings_shape():
    """OOF returns one embedding per TRAINING row, aligned to original order."""
    clf = _clf()
    X = np.random.RandomState(0).randn(20, 3)
    y = np.array([0, 1] * 10)

    oof = leave_one_fold_out_embeddings(clf, X, y, n_folds=5)

    assert oof.shape == (20, 16)


def test_leave_one_fold_out_refits_on_full_data():
    """After extraction the estimator is refit on the full training set, so it stays usable."""
    clf = _clf()
    X = np.random.RandomState(0).randn(20, 3)
    y = np.array([0, 1] * 10)

    leave_one_fold_out_embeddings(clf, X, y, n_folds=5)

    assert len(clf.X_train) == 20


def test_leave_one_fold_out_embeddings_regressor():
    """Works for the regressor too (plain KFold instead of StratifiedKFold)."""
    reg = _reg()
    X = np.random.RandomState(0).randn(20, 3)
    y = np.random.RandomState(1).randn(20)

    oof = leave_one_fold_out_embeddings(reg, X, y, n_folds=5)

    assert oof.shape == (20, 16)


def test_leave_one_fold_out_rejects_too_few_folds():
    """Fewer than 2 folds makes no sense for leave-one-fold-out."""
    clf = _clf()
    X = np.random.RandomState(0).randn(6, 3)
    y = np.array([0, 1, 0, 1, 0, 1])

    with pytest.raises(ValueError):
        leave_one_fold_out_embeddings(clf, X, y, n_folds=1)


def test_leave_one_fold_out_embeddings_mixed_dataframe():
    """Regression: OOF on a mixed-type DataFrame (string + numeric categoricals with NaN) must
    work AND leave the estimator usable for predict on a DataFrame afterwards. Previously
    np.asarray collapsed the frame to object dtype, poisoning the categorical encoder so a
    later predict raised "ufunc 'isnan' not supported".
    """
    torch.manual_seed(0)
    model = NanoTabPFNModel(
        embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=3
    )
    clf = NanoTabPFNClassifier(model=model, device="cpu", categorical_features=[1, 2], infer_categorical=False)

    rng = np.random.RandomState(0)
    n = 40
    X = pd.DataFrame({
        "num": rng.randn(n),                                 # continuous (col 0)
        "cat_str": rng.choice(["a", "b", "c"], size=n),       # string categorical (col 1)
        "cat_num": rng.randint(0, 3, size=n).astype(float),   # numeric low-card categorical (col 2)
    })
    X.loc[0, "num"] = np.nan
    X.loc[1, "cat_str"] = np.nan
    X.loc[2, "cat_num"] = np.nan
    y = pd.Series(["yes", "no"] * (n // 2))

    oof = leave_one_fold_out_embeddings(clf, X, y, n_folds=5)
    assert oof.shape == (n, 16)

    # The estimator must stay usable for prediction on a DataFrame (this is what regressed).
    preds = clf.predict(X)
    assert len(preds) == n