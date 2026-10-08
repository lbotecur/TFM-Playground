import numpy as np
import pytest
import torch

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.validation import PROBE_TEST_ROWS, prior_metrics, probe_data, probe_name


def _tiny_model():
    torch.manual_seed(0)
    return NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10)


def test_probe_data_shapes_labels_and_determinism():
    X_train, y_train, X_test, y_test = probe_data("xor3", 100, 20, seed=3)
    assert X_train.shape == (100, 23) and X_test.shape == (PROBE_TEST_ROWS, 23)
    assert set(np.unique(y_train)) == {0, 1}
    again = probe_data("xor3", 100, 20, seed=3)
    assert np.array_equal(X_train, again[0]) and np.array_equal(y_test, again[3])
    assert probe_name("xor3", 100, 20) == "xor3_n100_noise20"


def test_xor3_labels_are_the_sign_of_the_product_of_three_features():
    """Exactly three columns carry the signal (no noise added) and the label is their product's sign."""
    X_train, y_train, _, _ = probe_data("xor3", 200, 0, seed=0)
    assert np.array_equal(y_train, (np.prod(X_train, axis=1) > 0).astype(int))


def test_prior_metrics_on_handmade_tables():
    model = _tiny_model()
    tables = [{"x": torch.randn(20, 4), "y": torch.randint(0, 3, (20,)).float(), "split": 14} for _ in range(3)]
    metrics = prior_metrics(model, tables, "cpu", amp_dtype=None)
    assert set(metrics) == {"loss", "accuracy", "baseline_loss"}
    assert metrics["loss"] > 0 and 0 <= metrics["accuracy"] <= 1 and metrics["baseline_loss"] > 0


def test_prior_sets_are_built_once_and_reused(tmp_path, monkeypatch):
    """Tables of the requested size and split; widened sets get the extra features; the file is reused."""
    pytest.importorskip("tabicl")
    from tfmplayground import validation

    monkeypatch.setitem(validation.PRIOR_SETS, "tiny", dict(prior_type="graph_scm", rows=50, min_features=3,
                                                           max_features=5, widen=0))
    monkeypatch.setitem(validation.PRIOR_SETS, "tiny_wide", dict(prior_type="graph_scm", rows=50, min_features=3,
                                                                max_features=5, widen=30))
    sets = validation.load_prior_sets(tmp_path, n_tables=2, names=["tiny", "tiny_wide"])
    t = sets["tiny"][0]
    assert t["x"].shape[0] == 50 and t["x"].shape[1] <= 5 and t["split"] == 35
    assert sets["tiny_wide"][0]["x"].shape[1] >= 30
    before = (tmp_path / "tiny_n2.pt").stat().st_mtime
    again = validation.load_prior_sets(tmp_path, n_tables=2, names=["tiny"])
    assert (tmp_path / "tiny_n2.pt").stat().st_mtime == before
    assert torch.equal(again["tiny"][0]["x"], t["x"])


def test_prior_metrics_classifier_counts_only_classes_seen_in_training():
    """A test row of a class absent from the training rows is left out (no model can predict it); a model that
    separates the classes perfectly gets a loss near zero on the rest."""
    from tfmplayground.validation import prior_metrics_classifier

    rng = np.random.default_rng(0)
    y = torch.tensor([0, 1] * 20 + [2, 0, 1, 2], dtype=torch.float32)  # class 2 only among the test rows
    x = torch.tensor(rng.normal(size=(44, 1)) * 0.01, dtype=torch.float32) + 10 * y[:, None]
    tables = [{"x": x, "y": y, "split": 40}]
    metrics = prior_metrics_classifier("logreg", tables, "cpu")
    assert metrics["accuracy"] == 1.0 and metrics["loss"] < 0.1
