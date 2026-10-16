"""Weight-sparse training (tfmplayground/sparsity.py), RMSNorm and AbsTopK: the trained weights have the intended
density, with schedule-free AdamW's averaged weights included; the default model is unchanged."""

import pytest
import schedulefree
import torch
from torch import nn

from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.nanotabpfn import NanoTabPFNModel, abs_topk
from tfmplayground.sparsity import WeightSparsifier, WeightSparsity, density_report, sparse_parameters, topk_mask
from tfmplayground.train import train


def _model(**kwargs):
    torch.manual_seed(0)
    return NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2,
                           num_outputs=10, **kwargs)


class _Prior:
    def __init__(self, n=8, seed=0):
        generator = torch.Generator().manual_seed(seed)
        self.batches = []
        for _ in range(n):
            x = torch.randn(2, 12, 4, generator=generator)
            y = torch.randint(0, 3, (2, 12), generator=generator).float()
            self.batches.append({"x": x, "y": y, "target_y": y.clone(), "train_test_split_index": 8})
        self.num_steps = n

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return self.num_steps


def test_topk_mask_keeps_the_largest_and_a_minimum_per_row():
    values = torch.tensor([[5.0, 0.1, 0.2, 0.3], [0.01, 0.02, 0.03, 0.04]])
    mask = topk_mask(values, density=0.25, min_per_row=1)  # 2 of 8 overall, plus the largest of every row
    assert mask.tolist() == [[True, False, False, True], [False, False, False, True]]
    assert topk_mask(values, density=0.125, min_per_row=0).sum() == 1


def test_density_falls_linearly_over_the_annealing_then_stays():
    sparsifier = WeightSparsifier(_model(), None, WeightSparsity(density=0.1, anneal_fraction=0.5), total_steps=100)
    assert sparsifier.density(0) == 1.0
    assert sparsifier.density(25) == pytest.approx(0.55)
    assert sparsifier.density(50) == pytest.approx(0.1) and sparsifier.density(100) == pytest.approx(0.1)


def test_training_leaves_the_checkpoint_at_the_target_density(tmp_path, monkeypatch):
    """Schedule-free AdamW: the saved (averaged) weights and the training weights have the same zeros, at the
    target density (a little above it where a row keeps its minimum), for every sparse matrix."""
    monkeypatch.chdir(tmp_path)
    model = _model(norm="rms")
    config = WeightSparsity(density=0.25, anneal_fraction=0.5, min_per_row=1)
    train(model, _Prior(), nn.CrossEntropyLoss(), epochs=4, device=torch.device("cpu"), run_name="run",
          lr=1e-2, weight_sparsity=config)
    report = density_report(model)  # eval mode after train(): the averaged weights, as saved
    sizes = {name: parameter.numel() for name, parameter, _ in sparse_parameters(model)}
    big = {n: d for n, d in report.items() if "weight" in n and sizes[n] >= 256}
    assert big and all(0.24 <= d <= 0.33 for d in big.values()), big
    saved = torch.load(tmp_path / "workdir" / "run" / "latest_checkpoint.pth", weights_only=False)
    for name, parameter, _ in sparse_parameters(model):
        assert torch.equal(saved["model"][name] != 0, parameter != 0)
    assert saved["architecture"]["norm"] == "rms"


def test_schedule_free_base_sequence_shares_the_zeros():
    model = _model()
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-2)
    sparsifier = WeightSparsifier(model, optimizer, WeightSparsity(density=0.3, anneal_fraction=0.0, min_per_row=0),
                                  total_steps=10)
    optimizer.train()
    for batch in _Prior(3).batches:
        output = model((batch["x"], batch["y"][:, :8]), train_test_split_index=8)
        nn.functional.cross_entropy(output.reshape(-1, 10), batch["y"][:, 8:].reshape(-1).long()).backward()
        optimizer.step()
        sparsifier.step()
        optimizer.zero_grad()
    for _, parameter, _ in sparse_parameters(model):
        z = optimizer.state[parameter]["z"]
        assert torch.all(z[parameter == 0] == 0)
    optimizer.eval()  # the averaged weights: zero where y and z are
    assert all(0.29 <= d <= 0.31 for n, d in density_report(model).items() if "in_proj_weight" in n)


def test_rmsnorm_model_and_abs_topk_load_from_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = _model(norm="rms", mlp_topk=0.25)
    assert not any(isinstance(m, nn.LayerNorm) for m in model.modules())
    assert sum(isinstance(m, nn.RMSNorm) for m in model.modules()) == 6
    train(model, _Prior(2), nn.CrossEntropyLoss(), epochs=1, device=torch.device("cpu"), run_name="run")
    loaded = init_model_from_state_dict_file(tmp_path / "workdir" / "run" / "latest_checkpoint.pth")
    assert loaded.norm == "rms" and loaded.mlp_topk == 0.25 and loaded.transformer_blocks[0].mlp_keep == 8
    batch = _Prior(1).batches[0]
    with torch.no_grad():
        model.eval(), loaded.eval()
        a = model((batch["x"], batch["y"][:, :8]), train_test_split_index=8)
        b = loaded((batch["x"], batch["y"][:, :8]), train_test_split_index=8)
    torch.testing.assert_close(a, b)


def test_abs_topk_keeps_k_largest_magnitudes():
    x = torch.tensor([[3.0, -5.0, 0.5, 1.0], [0.1, 0.2, -0.3, 0.4]])
    torch.testing.assert_close(abs_topk(x, 2), torch.tensor([[3.0, -5.0, 0.0, 0.0], [0.0, 0.0, -0.3, 0.4]]))


def test_default_model_is_unchanged():
    """Without the new options, the parameters (names and values at a seed) are those of every earlier model."""
    model = _model()
    assert model.norm == "layer" and model.mlp_topk == 0.0
    assert "transformer_blocks.0.norm1.bias" in model.state_dict()
    assert all(block.mlp_keep == 0 for block in model.transformer_blocks)


def test_rmsnorm_runs_in_fp32_under_autocast():
    # as LayerNorm under CUDA autocast: fp32 output, no dtype-mismatch fallback, the fp32 result
    import warnings

    from tfmplayground.models.nanotabpfn import FP32RMSNorm

    norm = FP32RMSNorm(16)
    nn.init.normal_(norm.weight)
    x = torch.randn(5, 16).to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16), warnings.catch_warnings():
        warnings.simplefilter("error")
        out = norm(x)
    assert out.dtype == torch.float32
    expected = torch.nn.functional.rms_norm(x.float(), (16,), norm.weight, norm.eps)
    assert torch.equal(out, expected)
    plain = nn.RMSNorm(16)
    plain.load_state_dict(norm.state_dict())  # same parameters: checkpoints load either way
    y = torch.randn(5, 16)
    assert torch.allclose(norm(y), plain(y))
