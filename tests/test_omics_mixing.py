"""MixedOmicsPrior (tfmplayground/priors/omics_mechanisms_loader.py) mixes omics tables into a base prior's batches
without changing its length or its batches, and train() does not widen the omics tables again."""

import pytest
import torch
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors.omics_mechanisms import MechanismPriorConfig
from tfmplayground.priors.omics_mechanisms_loader import MixedOmicsPrior

SMALL = MechanismPriorConfig(min_rows=40, max_rows=60, max_total_features=30, num_other_mechanisms=(0, 2),
                             num_independent=(0, 3))  # fmt: skip


class Base:
    """A base prior of 6 fixed batches, counting its epochs as TabICLPriorDataLoader does."""

    def __init__(self):
        x, y = torch.randn(1, 40, 3), torch.randint(0, 3, (1, 40)).float()
        self.batches = [{"x": x, "y": y, "target_y": y.clone(), "train_test_split_index": 30} for _ in range(6)]
        self.epoch = 0

    def __iter__(self):
        self.epoch += 1
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def test_fraction_zero_is_the_base_prior():
    base = Base()
    mixed = MixedOmicsPrior(base, 0.0, SMALL, seed=0)
    assert len(mixed) == 6 and all(a is b for a, b in zip(mixed, base.batches, strict=True))


def test_fraction_one_gives_omics_batches_marked_as_widened():
    mixed = MixedOmicsPrior(Base(), 1.0, SMALL, seed=0)
    batches = list(mixed)
    assert len(batches) == 6
    for b in batches:
        assert b["pre_widened"] and b["x"].dim() == 3 and b["x"].shape[0] == 1
        rows = b["x"].shape[1]
        assert 40 <= rows <= 60 and 1 <= b["train_test_split_index"] < rows
        assert b["y"].shape == (1, rows) and torch.equal(b["y"], b["target_y"])
        assert b["y"].min() >= 0 and b["y"].max() < 10


def test_mixing_is_seeded_per_epoch_and_follows_a_resume():
    first = [b.get("pre_widened", False) for b in MixedOmicsPrior(Base(), 0.5, SMALL, seed=3)]
    again = [b.get("pre_widened", False) for b in MixedOmicsPrior(Base(), 0.5, SMALL, seed=3)]
    assert first == again and any(first) and not all(first)
    mixed = MixedOmicsPrior(Base(), 0.5, SMALL, seed=3)
    list(mixed)
    second_epoch = [b.get("pre_widened", False) for b in mixed]
    resumed = MixedOmicsPrior(Base(), 0.5, SMALL, seed=3)
    resumed.epoch = 1  # as pretrain.py does for a --resume after one epoch
    assert [b.get("pre_widened", False) for b in resumed] == second_epoch


def test_fraction_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        MixedOmicsPrior(Base(), 1.5, SMALL)


def test_train_does_not_widen_omics_batches(tmp_path, monkeypatch):
    import tfmplayground.train as train_module
    from tfmplayground.train import WideningConfig, train

    monkeypatch.chdir(tmp_path)
    widened = []
    real = train_module.add_mixed_widening_features
    monkeypatch.setattr(train_module, "add_mixed_widening_features",
                        lambda x, *a, **k: widened.append(x.shape) or real(x, *a, **k))  # fmt: skip
    torch.manual_seed(0)
    model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10)
    prior = MixedOmicsPrior(Base(), 0.5, SMALL, seed=1)
    omics = sum(b.get("pre_widened", False) for b in MixedOmicsPrior(Base(), 0.5, SMALL, seed=1))
    config = WideningConfig(add_features_min=5, add_features_max=5, prob_no_widening=0.0)  # widen every batch
    train(model, prior, nn.CrossEntropyLoss(), epochs=1, device=torch.device("cpu"), run_name="run", widening=config)
    assert 0 < omics < 6 and len(widened) == 6 - omics  # only the base batches were widened
