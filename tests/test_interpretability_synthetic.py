import numpy as np
import torch
from sklearn.datasets import make_classification

from tfmplayground.interpretability import feature_importance, fit_and_explain, informative_auroc, recall_at_k
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def test_recall_at_k_and_informative_auroc():
    scores = np.array([0.9, 0.1, 0.8, 0.2, np.nan])
    assert recall_at_k(scores, [0, 2]) == 1.0
    assert recall_at_k(scores, [0, 1]) == 0.5
    assert informative_auroc(scores, [0, 2]) == 1.0
    assert informative_auroc(scores, [1, 4]) == 0.0  # NaN (dropped column) ranks last


def test_random_forest_finds_the_informative_features_of_an_easy_problem():
    X, y = make_classification(n_samples=200, n_features=30, n_informative=3, n_redundant=0, n_repeated=0,
                               shuffle=False, class_sep=3.0, random_state=0)
    scores = feature_importance("random_forest", X, y, device="cpu")
    assert scores.shape == (30,) and recall_at_k(scores, np.arange(3)) == 1.0


def test_our_checkpoint_gives_one_attention_score_per_feature(tmp_path):
    torch.manual_seed(0)
    model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10)
    arch = {"num_layers": 2, "embedding_size": 16, "num_attention_heads": 2, "mlp_hidden_size": 32, "num_outputs": 10}
    torch.save({"architecture": arch, "model": model.state_dict()}, tmp_path / "m.pth")
    X, y = make_classification(n_samples=50, n_features=20, n_informative=3, n_redundant=0, random_state=0)
    scores = feature_importance(str(tmp_path / "m.pth"), X, y, device="cpu", amp_dtype=None)
    assert scores.shape == (20,) and np.all(np.isfinite(scores)) and np.all(scores > 0)


def test_fit_and_explain_returns_scores_and_test_probabilities(tmp_path):
    torch.manual_seed(0)
    model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10)
    arch = {"num_layers": 2, "embedding_size": 16, "num_attention_heads": 2, "mlp_hidden_size": 32, "num_outputs": 10}
    torch.save({"architecture": arch, "model": model.state_dict()}, tmp_path / "m.pth")
    X, y = make_classification(n_samples=80, n_features=20, n_informative=3, n_redundant=0, random_state=0)
    for name in (str(tmp_path / "m.pth"), "random_forest"):
        scores, proba = fit_and_explain(name, X[:50], y[:50], X[50:], device="cpu", amp_dtype=None)
        assert scores.shape == (20,) and proba.shape == (30, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-5)
