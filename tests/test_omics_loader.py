import torch
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors.omics import OmicsPriorConfig
from tfmplayground.priors.omics_loader import OmicsPriorDataLoader
from tfmplayground.train import train


def _loader(num_steps=3, batch_size=2, seed=0, **config):
    config = OmicsPriorConfig(**{"max_total_features": 200, **config})
    return OmicsPriorDataLoader(num_steps, batch_size, torch.device("cpu"), config=config, seed=seed)


def test_batches_have_the_train_format():
    loader = _loader(num_steps=5)
    batches = list(loader)
    assert len(batches) == len(loader) == loader.num_steps == 5
    for b in batches:
        x, y = b["x"], b["y"]
        assert x.dtype == y.dtype == torch.float32
        assert x.shape[:2] == y.shape == (2, x.shape[1])  # (batch, rows, features), shared rows
        assert torch.isfinite(x).all()
        assert torch.equal(b["target_y"], y)
        assert torch.equal(y, y.round()) and y.min() >= 0 and y.max() <= 9
        assert 0.3 * x.shape[1] - 1 <= b["train_test_split_index"] <= 0.9 * x.shape[1] + 1


def test_rows_and_width_follow_the_config():
    for b in _loader(num_steps=10, min_rows=50, max_rows=60, max_total_features=300):
        assert 50 <= b["x"].shape[1] <= 60
        assert b["x"].shape[2] <= 300


def test_same_seed_same_batches():
    a, b = list(_loader(seed=3)), list(_loader(seed=3))
    for ba, bb in zip(a, b, strict=True):
        assert torch.equal(ba["x"], bb["x"]) and torch.equal(ba["y"], bb["y"])
        assert ba["train_test_split_index"] == bb["train_test_split_index"]


def test_train_runs_on_the_omics_prior(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10)
    _, loss = train(
        model,
        _loader(num_steps=2),
        nn.CrossEntropyLoss(),
        epochs=1,
        accumulate_gradients=2,
        device=torch.device("cpu"),
        run_name="run",
        missing_rate_max=0.1,
    )
    assert torch.isfinite(torch.tensor(loss))
