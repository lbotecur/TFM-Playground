"""graph_scm_raw: the same tables as TabICL's graph_scm without its clamp and z-score, plus categorical masks."""

import math
import random

import numpy as np
import pytest
import torch

pytest.importorskip("tabicl")
from tabicl.prior._dataset import GraphPrior  # noqa: E402
from tabicl.prior._reg2cls import outlier_removing, standard_scaling  # noqa: E402

from tfmplayground.augmentation import add_mixed_widening_features, recode_categories  # noqa: E402
from tfmplayground.external_priors.graph_raw import RawGraphPrior  # noqa: E402
from tfmplayground.external_priors.tabicl import TabICLPriorDataLoader  # noqa: E402

KWARGS = dict(batch_size=6, batch_size_per_gp=2, min_features=2, max_features=30, max_classes=10,
              min_seq_len=40, max_seq_len=200, log_seq_len=True, min_train_size=0.3, max_train_size=0.9, n_jobs=1)


def seeded(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batch(prior_cls, seed, **extra):
    seeded(seed)
    return prior_cls(**KWARGS, **extra).get_batch()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_not_raw_is_tabicl_bit_for_bit(seed):
    ours = batch(RawGraphPrior, seed, raw=False)
    theirs = batch(GraphPrior, seed)
    assert len(ours) == 6
    for a, b in zip(ours[:5], theirs):
        assert torch.equal(a, b)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_raw_is_tabicl_before_clamp_and_zscore(seed):
    raw = batch(RawGraphPrior, seed, raw=True)
    theirs = batch(GraphPrior, seed)
    X_raw, y_raw, d_raw = raw[:3]
    X, y, d = theirs[:3]
    assert torch.equal(y_raw, y) and torch.equal(d_raw, d)  # same graphs, labels and kept columns
    for i in range(X.shape[0]):
        cols = X_raw[i, :, : d[i]]
        assert torch.allclose(standard_scaling(outlier_removing(cols, threshold=4)), X[i, :, : d[i]], atol=1e-5)
        assert torch.all(X_raw[i, :, d[i]:] == 0)  # padding


def test_categorical_mask_marks_codes():
    X, _, d, _, _, masks = batch(RawGraphPrior, 3, raw=True)
    n_categorical = n_numeric = 0
    for i in range(X.shape[0]):
        assert not masks[i, d[i]:].any()  # no categorical padding
        for j in range(int(d[i])):
            col = X[i, :, j]
            if masks[i, j]:
                n_categorical += 1
                assert torch.equal(col, col.round()) and col.min() >= 0  # codes 0..K-1
            else:
                n_numeric += 1
    assert n_categorical > 0 and n_numeric > 0


def test_loader_hands_masks_to_training():
    loader = TabICLPriorDataLoader(num_steps=2, batch_size=4, num_datapoints_min=40, num_datapoints_max=120,
                                   min_features=2, max_features=20, max_num_classes=5, device=torch.device("cpu"),
                                   prior_type="graph_scm_raw", seed=0)
    for data in loader:
        assert data["categorical_mask"].shape == (data["x"].shape[0], data["x"].shape[2])
        assert data["categorical_mask"].dtype == torch.bool


def test_recode_categories():
    col = torch.tensor([2.5, -1.0, 2.5, float("nan"), 7.0, -1.0])
    out = recode_categories(col, torch.Generator().manual_seed(0))
    assert torch.isnan(out[3])
    present = out[~torch.isnan(out)]
    assert sorted(present.unique().tolist()) == [0.0, 1.0, 2.0]
    assert out[0] == out[2] and out[1] == out[5] and len({out[0].item(), out[1].item(), out[4].item()}) == 3


def test_widening_uses_given_masks_and_recodes():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 50, 4)  # four continuous columns: detection would call none categorical
    masks = torch.tensor([[True, True, False, False]])
    out = add_mixed_widening_features(x, 10, sparsity=0.5, noise_std=0.0, include_original_prob=0.0,
                                      categorical_masks=masks, recode_categoricals=True, generator=g)
    # half of the budget goes to categorical columns (the last block), with the codes 0..K-1
    continuous, categorical = out[0, :, :5], out[0, :, 5:]
    assert not torch.equal(continuous, continuous.round())
    for c in categorical.T:
        assert c.unique().tolist() == list(range(len(c.unique())))


def test_train_runs_on_raw_prior_with_widening(tmp_path, monkeypatch):
    """End to end: graph_scm_raw batches (with their masks) through train() with widening."""
    from torch import nn

    from tfmplayground.models.nanotabpfn import NanoTabPFNModel
    from tfmplayground.train import WideningConfig, train

    monkeypatch.chdir(tmp_path)
    loader = TabICLPriorDataLoader(num_steps=2, batch_size=2, num_datapoints_min=40, num_datapoints_max=80,
                                   min_features=2, max_features=10, max_num_classes=5, device=torch.device("cpu"),
                                   prior_type="graph_scm_raw", seed=0)
    torch.manual_seed(0)
    model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2,
                            num_outputs=10)
    _, total_loss = train(model, loader, nn.CrossEntropyLoss(), epochs=1, device=torch.device("cpu"),
                          run_name="run", widening=WideningConfig(add_features_min=20, add_features_max=20,
                                                                  sparsity_max=0.1, noise_max=0.5))
    assert math.isfinite(total_loss) and total_loss > 0
