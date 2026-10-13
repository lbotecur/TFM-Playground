"""Optional fixed-seed column permutation at inference (off for checkpoints without the setting, as the v2)."""

import numpy as np
import torch

from tfmplayground.interface import NanoTabPFNClassifier, export_checkpoint, init_model_from_state_dict_file
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def _model():
    torch.manual_seed(0)
    return NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2,
                           num_outputs=10)


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 7)).astype(np.float32)
    return X, (X[:, 0] + X[:, 3] > 0).astype(int)


def _clf(**kwargs):
    return NanoTabPFNClassifier(model=_model(), device="cpu", **kwargs)


def test_off_by_default_keeps_the_preprocessor_order():
    X, y = _data()
    default, off = _clf().fit(X[:60], y[:60]), _clf(permute_columns=False).fit(X[:60], y[:60])
    assert np.array_equal(default.column_order_, np.arange(7))
    assert np.array_equal(default.predict_proba(X[60:]), off.predict_proba(X[60:]))


def test_permutation_is_fixed_by_the_seed():
    X, y = _data()
    a = _clf(permute_columns=True).fit(X[:60], y[:60])
    b = _clf(permute_columns=True).fit(X[:60], y[:60])
    c = _clf(permute_columns=True, column_seed=1).fit(X[:60], y[:60])
    assert sorted(a.column_order_) == list(range(7)) and not np.array_equal(a.column_order_, np.arange(7))
    assert np.array_equal(a.column_order_, b.column_order_) and not np.array_equal(a.column_order_, c.column_order_)
    assert np.array_equal(a.predict_proba(X[60:]), b.predict_proba(X[60:]))


def test_same_as_feeding_the_permuted_columns_and_attention_maps_back():
    """With the permutation, the model sees X[:, order]: predictions equal those of the unpermuted classifier on
    X[:, order], and each attention score lands on its original column."""
    X, y = _data()
    permuted = _clf(permute_columns=True).fit(X[:60], y[:60])
    order = permuted.column_order_
    plain = _clf(permute_columns=False).fit(X[:60, order], y[:60])
    assert np.allclose(permuted.predict_proba(X[60:]), plain.predict_proba(X[60:, order]), atol=1e-6)
    scores_permuted = permuted.feature_attention_scores(X[60:])
    scores_plain = plain.feature_attention_scores(X[60:, order])  # score j belongs to original column order[j]
    assert np.allclose(scores_permuted[order], scores_plain, atol=1e-6)


def test_categorical_columns_no_longer_all_last():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.integers(0, 3, 80), rng.normal(size=(80, 5)), rng.integers(0, 2, 80)]).astype(float)
    y = np.arange(80) % 2
    clf = _clf(permute_columns=True, categorical_features=[0, 6], infer_categorical=False).fit(X[:60], y[:60])
    # preprocessed: 5 numeric then 2 categorical (positions 5 and 6); after the permutation not both at the end
    assert set(clf.column_order_[-2:]) != {5, 6}
    assert clf.predict_proba(X[60:]).shape == (20, 2)


def test_shared_helpers_order_train_and_test_rows_alike():
    """The classifier and the regressor share _fit_columns/_transform_columns."""
    from tfmplayground import interface

    class Estimator:
        categorical_features, infer_categorical = None, True
        max_unique_for_categorical, min_samples_for_categorical_inference = 10, 30
        permute_columns, column_seed = True, 0

    X, _ = _data()
    est = Estimator()
    assert np.array_equal(interface._fit_columns(est, X[:60]), X[:60][:, est.column_order_])
    assert np.array_equal(interface._transform_columns(est, X[60:]), X[60:][:, est.column_order_])


def test_export_stores_the_setting_and_loading_uses_it(tmp_path):
    source, release = tmp_path / "ckpt.pth", tmp_path / "release.pth"
    model = _model()
    architecture = dict(num_attention_heads=2, embedding_size=16, mlp_hidden_size=32, num_layers=2, num_outputs=10,
                        feature_clip=10.0)
    torch.save({"architecture": architecture, "model": model.state_dict(), "epoch": 1}, source)
    assert init_model_from_state_dict_file(str(source)).permute_columns is False  # as the v2
    assert export_checkpoint(str(source), str(release), permute_columns=True)["permute_columns"] is True
    clf = NanoTabPFNClassifier(model=str(release), device="cpu")
    assert clf.permute_columns is True
    assert NanoTabPFNClassifier(model=str(release), device="cpu", permute_columns=False).permute_columns is False
