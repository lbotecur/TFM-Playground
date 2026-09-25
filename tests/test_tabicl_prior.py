import pytest
import torch


def test_tabicl_prior_loader_yields_batch():
    """Importing and iterating the TabICL loader must work (regression for the outdated
    tabicl.prior.dataset import path)."""
    pytest.importorskip("tabicl")
    from tfmplayground.external_priors import TabICLPriorDataLoader

    loader = TabICLPriorDataLoader(
        num_steps=1, batch_size=1,
        num_datapoints_min=50, num_datapoints_max=100,
        min_features=5, max_features=10,
        max_num_classes=3, device=torch.device("cpu"),
    )
    batch = next(iter(loader))
    assert "x" in batch and "y" in batch