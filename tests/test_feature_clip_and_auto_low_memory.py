import numpy as np
import pytest
import torch
from torch import nn

import tfmplayground.interface as interface
from tfmplayground.interface import NanoTabPFNClassifier, export_checkpoint, init_model_from_state_dict_file
from tfmplayground.models.nanotabpfn import NanoTabPFNModel, TransformerEncoderLayer, normalize_features
from tfmplayground.train import train


def _model(**kwargs):
    torch.manual_seed(0)
    return NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2,
                           num_outputs=10, **kwargs)


def _data(extreme: bool):
    """Gaussian features; with extreme=True, two sparse columns: a single nonzero value in the 60 training
    rows (z ~ 7.7) and larger values in some test rows (z ~ 38), as in bag-of-words text."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 8)).astype(np.float32)
    if extreme:
        X[:, :2] = 0.0
        X[0, 0], X[[62, 65, 71], 0] = 1.0, 5.0
        X[1, 1], X[[60, 77], 1] = 1.0, 5.0
    return X, np.arange(80) % 2


def _proba(X, y, **kwargs):
    """Fits on the first 60 rows and predicts the last 20, with a fresh copy of the same model."""
    return NanoTabPFNClassifier(model=_model(), device="cpu", **kwargs).fit(X[:60], y[:60]).predict_proba(X[60:])


# --- feature clip -------------------------------------------------------------------------------


def test_normalize_features_clip_bounds_the_values_and_defaults_to_10():
    x = torch.tensor([[[0.0], [1.0], [0.0], [-1.0], [1e6]]])  # last row is a test row
    assert normalize_features(x, 4)[0, -1, 0, 0].item() == 10.0
    assert normalize_features(x, 4, clip=100.0)[0, -1, 0, 0].item() == 100.0
    # the missing indicator channel is not clipped or changed
    torch.testing.assert_close(normalize_features(x, 4, clip=100.0)[..., 1], normalize_features(x, 4)[..., 1])


def test_model_feature_clip_defaults_to_10_and_is_not_a_weight():
    assert _model().feature_encoder.clip == 10.0
    model = _model(feature_clip=100.0)
    assert model.feature_encoder.clip == 100.0
    assert set(model.state_dict()) == set(_model().state_dict())  # same weights: old checkpoints still load


def test_checkpoint_keeps_the_feature_clip_and_old_checkpoints_load_with_100(tmp_path):
    model = _model(feature_clip=10.0)
    arch = {"num_layers": 2, "embedding_size": 16, "num_attention_heads": 2, "mlp_hidden_size": 32, "num_outputs": 10}
    torch.save({"architecture": {**arch, "feature_clip": 10.0}, "model": model.state_dict()}, tmp_path / "new.pth")
    torch.save({"architecture": arch, "model": model.state_dict()}, tmp_path / "old.pth")
    assert init_model_from_state_dict_file(str(tmp_path / "new.pth")).feature_encoder.clip == 10.0
    assert init_model_from_state_dict_file(str(tmp_path / "old.pth")).feature_encoder.clip == 100.0


def test_train_saves_the_feature_clip_in_the_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    x = torch.randn(1, 8, 3)
    y = torch.randint(0, 10, (1, 8)).float()

    class Prior(list):
        num_steps = 1

    prior = Prior([{"x": x, "y": y, "target_y": y.clone(), "train_test_split_index": 5}])
    train(_model(feature_clip=10.0), prior, nn.CrossEntropyLoss(), epochs=1, device=torch.device("cpu"), run_name="r")
    state = torch.load(tmp_path / "workdir" / "r" / "latest_checkpoint.pth", weights_only=False)
    assert state["architecture"]["feature_clip"] == 10.0


def test_classifier_feature_clip_overrides_the_checkpoint_and_none_keeps_it():
    assert NanoTabPFNClassifier(model=_model(feature_clip=100.0), device="cpu").model.feature_encoder.clip == 100.0
    clf = NanoTabPFNClassifier(model=_model(feature_clip=100.0), device="cpu", feature_clip=10.0)
    assert clf.model.feature_encoder.clip == 10.0


def test_export_checkpoint_sets_the_feature_clip_and_drops_the_optimizer(tmp_path):
    """The release copy of an old checkpoint (no feature_clip recorded, so 100) loads with the declared clip and
    the same weights, and gives the same predictions as setting the clip by hand."""
    model = _model(feature_clip=100.0)
    arch = {"num_layers": 2, "embedding_size": 16, "num_attention_heads": 2, "mlp_hidden_size": 32, "num_outputs": 10}
    torch.save({"epoch": 30, "architecture": arch, "model": model.state_dict(), "optimizer": {"big": 1}},
               tmp_path / "epoch_30.pth")
    assert export_checkpoint(str(tmp_path / "epoch_30.pth"), str(tmp_path / "v2.pth"), 10.0)["feature_clip"] == 10.0
    state = torch.load(tmp_path / "v2.pth", weights_only=False)
    assert "optimizer" not in state and state["epoch"] == 30
    released = init_model_from_state_dict_file(str(tmp_path / "v2.pth"))
    assert released.feature_encoder.clip == 10.0
    X, y = _data(extreme=True)
    kwargs = {"device": "cpu", "categorical_features": [], "infer_categorical": False}
    by_file = NanoTabPFNClassifier(model=released, **kwargs).fit(X[:60], y[:60]).predict_proba(X[60:])
    old = init_model_from_state_dict_file(str(tmp_path / "epoch_30.pth"))
    assert old.feature_encoder.clip == 100.0
    by_hand = NanoTabPFNClassifier(model=old, feature_clip=10.0, **kwargs).fit(X[:60], y[:60]).predict_proba(X[60:])
    np.testing.assert_array_equal(by_file, by_hand)
    # without a clip, the export keeps the source's own
    assert export_checkpoint(str(tmp_path / "epoch_30.pth"), str(tmp_path / "same.pth"))["feature_clip"] == 100.0


def test_feature_clip_changes_nothing_when_no_value_reaches_it():
    X, y = _data(extreme=False)
    assert np.abs((X - X[:60].mean(0)) / X[:60].std(0, ddof=1)).max() < 10
    p100 = _proba(X, y, feature_clip=100.0)
    p10 = _proba(X, y, feature_clip=10.0)
    np.testing.assert_array_equal(p10, p100)


def test_feature_clip_changes_the_predictions_when_values_exceed_it():
    X, y = _data(extreme=True)
    # declared numeric: otherwise the sparse columns are inferred categorical and the unseen test values imputed
    kwargs = {"device": "cpu", "categorical_features": [], "infer_categorical": False}
    p100 = NanoTabPFNClassifier(model=_model(), feature_clip=100.0, **kwargs).fit(X[:60], y[:60]).predict_proba(X[60:])
    p10 = NanoTabPFNClassifier(model=_model(), feature_clip=10.0, **kwargs).fit(X[:60], y[:60]).predict_proba(X[60:])
    changed = ~np.isclose(p10, p100).all(axis=1)
    # only the test rows with a clipped value (rows 60, 62, 65, 71, 77) can change
    assert changed.any() and set(np.where(changed)[0] + 60) <= {60, 62, 65, 71, 77}


# --- automatic low-memory inference ---------------------------------------------------------------


def _low_memory_flags(clf):
    return {block.low_memory for block in clf.model.transformer_blocks}


def test_low_memory_is_the_default():
    clf = NanoTabPFNClassifier(model=_model(), device="cpu")
    assert clf.low_memory is True and _low_memory_flags(clf) == {True}
    X, y = _data(extreme=False)
    np.testing.assert_array_equal(clf.fit(X[:60], y[:60]).predict_proba(X[60:]), _proba(X, y, low_memory=True))


def test_auto_keeps_the_normal_path_for_small_tables():
    X, y = _data(extreme=False)
    clf = NanoTabPFNClassifier(model=_model(), device="cpu", low_memory="auto")
    auto = clf.fit(X[:60], y[:60]).predict_proba(X[60:])
    assert _low_memory_flags(clf) == {False}
    normal = _proba(X, y, low_memory=False)
    np.testing.assert_array_equal(auto, normal)


def test_auto_switches_to_low_memory_for_large_tables(monkeypatch):
    X, y = _data(extreme=False)
    # 80 rows x (8 + 1) columns x 16 = 11520 elements
    monkeypatch.setattr(interface, "LOW_MEMORY_AUTO_ELEMENTS", 11520)
    clf = NanoTabPFNClassifier(model=_model(), device="cpu", low_memory="auto")
    auto = clf.fit(X[:60], y[:60]).predict_proba(X[60:])
    assert _low_memory_flags(clf) == {True}
    forced = _proba(X, y, low_memory=True)
    np.testing.assert_array_equal(auto, forced)
    monkeypatch.setattr(interface, "LOW_MEMORY_AUTO_ELEMENTS", 11521)
    clf.predict_proba(X[60:])
    assert _low_memory_flags(clf) == {False}


def test_low_memory_gives_the_normal_results_in_fp32():
    X, y = _data(extreme=True)
    normal = _proba(X, y, low_memory=False)
    low = _proba(X, y, low_memory=True)
    np.testing.assert_allclose(low, normal, rtol=1e-5, atol=1e-6)


def _oom_in_normal_mode(monkeypatch):
    real = TransformerEncoderLayer.forward

    def forward(self, *args, **kwargs):
        if not self.low_memory:
            raise torch.cuda.OutOfMemoryError("simulated")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(TransformerEncoderLayer, "forward", forward)


def test_auto_retries_in_low_memory_after_an_out_of_memory_error(monkeypatch):
    X, y = _data(extreme=False)
    expected = _proba(X, y, low_memory=True)
    _oom_in_normal_mode(monkeypatch)
    clf = NanoTabPFNClassifier(model=_model(), device="cpu", low_memory="auto").fit(X[:60], y[:60])
    with pytest.warns(UserWarning, match="low-memory"):
        proba = clf.predict_proba(X[60:])
    np.testing.assert_array_equal(proba, expected)


def test_forced_normal_mode_does_not_hide_an_out_of_memory_error(monkeypatch):
    X, y = _data(extreme=False)
    _oom_in_normal_mode(monkeypatch)
    clf = NanoTabPFNClassifier(model=_model(), device="cpu", low_memory=False).fit(X[:60], y[:60])
    with pytest.raises(torch.cuda.OutOfMemoryError):
        clf.predict_proba(X[60:])


def test_invalid_low_memory_value_is_rejected():
    with pytest.raises(ValueError, match="low_memory"):
        NanoTabPFNClassifier(model=_model(), device="cpu", low_memory="yes")


def test_feature_attention_and_embeddings_work_in_auto_low_memory(monkeypatch):
    X, y = _data(extreme=False)
    normal = NanoTabPFNClassifier(model=_model(), device="cpu", low_memory=False).fit(X[:60], y[:60])
    scores, emb = normal.feature_attention_scores(X[60:]), normal.get_embeddings(X[60:])
    monkeypatch.setattr(interface, "LOW_MEMORY_AUTO_ELEMENTS", 1)
    auto = NanoTabPFNClassifier(model=_model(), device="cpu", low_memory="auto").fit(X[:60], y[:60])
    np.testing.assert_allclose(auto.feature_attention_scores(X[60:]), scores, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(auto.get_embeddings(X[60:]), emb, rtol=1e-4, atol=1e-5)
    assert _low_memory_flags(auto) == {True}
