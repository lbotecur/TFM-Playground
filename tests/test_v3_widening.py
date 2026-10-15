import math

import numpy as np
import torch
from torch import nn

from tfmplayground.augmentation import sparsify_columns
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.train import WideningConfig, shorten_table, train


def test_shorten_table_keeps_the_share_of_training_rows_and_aligned_labels():
    x = torch.arange(20).float().view(1, 20, 1).repeat(2, 1, 3)  # row index as the value
    y = torch.arange(20).float().view(1, 20).repeat(2, 1)
    xs, ys, ts, split = shorten_table(x, y, y.clone(), split=12, max_rows=10)
    assert xs.shape == (2, 10, 3) and split == 6  # 12/20 of 10 rows
    kept = xs[0, :, 0].long().tolist()
    assert kept == [0, 1, 2, 3, 4, 5, 12, 13, 14, 15]  # first training rows, then first test rows
    assert torch.equal(ys[0], xs[0, :, 0]) and torch.equal(ts[0], xs[0, :, 0])


def test_shorten_table_keeps_at_least_two_training_rows_and_one_test_row():
    x, y = torch.randn(1, 100, 2), torch.zeros(1, 100)
    _, _, _, split = shorten_table(x, y, y, split=2, max_rows=10)
    assert split == 2
    xs, _, _, split = shorten_table(x, y, y, split=99, max_rows=10)
    assert split == 9 and xs.shape[1] == 10


def test_sparsify_columns_makes_rare_nonnegative_values_and_keeps_missing_ones():
    torch.manual_seed(0)
    x = torch.randn(2, 3000, 40)
    x[0, :7, 5] = float("nan")
    g = torch.Generator().manual_seed(0)
    out = sparsify_columns(x, 1.0, generator=g)
    observed = ~torch.isnan(out)
    nonzero = ((out != 0) & observed).float().sum(1) / observed.float().sum(1)
    assert nonzero.min() >= 0.0015 and nonzero.max() <= 0.21  # share drawn in [0.002, 0.2]
    assert (out[observed] >= 0).all() and torch.isnan(out).sum() == 7
    # a nonzero entry is one of the column's highest values: order is kept among them
    col = x[1, :, 0]
    kept = out[1, :, 0] > 0
    assert col[kept].min() >= col[~kept].max()
    assert torch.equal(sparsify_columns(x, 0.0).nan_to_num(), x.nan_to_num())


def test_training_with_a_row_cap_and_sparse_columns_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10)
    x = torch.randn(1, 40, 3)
    y = torch.randint(0, 3, (1, 40)).float()

    class Prior(list):
        num_steps = 2

    prior = Prior([{"x": x, "y": y, "target_y": y.clone(), "train_test_split_index": 25}] * 2)
    seen = []
    real = NanoTabPFNModel.forward

    def spy(self, *args, **kwargs):
        seen.append(args[0][0].shape)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(NanoTabPFNModel, "forward", spy)
    widening = WideningConfig(add_features_min=20, add_features_max=20, max_rows=16, sparse_columns_prob=1.0)
    _, loss = train(model, prior, nn.CrossEntropyLoss(), epochs=1, device=torch.device("cpu"), run_name="r",
                    widening=widening)
    assert np.isfinite(loss) and all(shape[1] == 16 for shape in seen)


def test_run_dev_reads_the_hdlss_development_datasets(tmp_path):
    from scipy.io import savemat

    from tfmplayground.benchmarks import hdlss
    from tfmplayground.validation import run_dev

    rng = np.random.default_rng(0)
    y = np.arange(60) % 2 + 1
    X = rng.normal(size=(60, 30)) + y[:, None] * (np.arange(30) < 3)
    (tmp_path / hdlss.FOLDER).mkdir()
    savemat(tmp_path / hdlss.FOLDER / "RELATHE.mat", {"X": X, "Y": y.reshape(-1, 1)})
    rows = run_dev("logreg", "cpu", tmp_path, datasets=("RELATHE", "madelon"))  # madelon missing: skipped
    assert {r["set"] for r in rows} == {"dev_RELATHE"} and len(rows) == 2 * 6  # 3-fold x 2 repeats, 2 metrics


# --- cell budget and ramp (experiments/memory_budget.py: ~4M cells per table fit on 80 GB) ---------------------

def test_budget_shapes_are_log_uniform_and_fit_the_budget():
    from tfmplayground.train import WideningConfig, budget_shape

    torch.manual_seed(0)
    config = WideningConfig(cell_budget=3_500_000, width_min=200, width_max=10_000, min_rows=40)
    shapes = [budget_shape(config, 10_000) for _ in range(4000)]
    widths = torch.tensor([w for w, _ in shapes], dtype=torch.float)
    assert all(200 <= w <= 10_000 and 3 <= r <= min(10_000, 3_500_000 // w) for w, r in shapes)
    assert abs(widths.log().median().exp().item() / (200 * 10_000) ** 0.5 - 1) < 0.1  # log-uniform: geometric mean
    assert abs((widths < 1000).float().mean().item() - math.log(5) / math.log(50)) < 0.03
    assert min(r for _, r in shapes) <= 60  # short tables of every width, not only the cap
    # a short prior table is not lengthened
    assert all(r <= 300 for _, r in (budget_shape(config, 300) for _ in range(200)))


def test_widening_ramp_starts_narrow_and_reaches_the_share():
    from tfmplayground.train import WideningConfig, widening_threshold

    config = WideningConfig(prob_no_widening=0.7, ramp_epochs=10)
    assert widening_threshold(config, 1) == 1.0  # epoch 1: no widened batch
    assert abs(widening_threshold(config, 6) - (1 - 0.3 * 0.5)) < 1e-12
    assert widening_threshold(config, 11) == 0.7 and widening_threshold(config, 50) == 0.7
    assert widening_threshold(WideningConfig(prob_no_widening=0.7), 1) == 0.7  # no ramp: as before


def test_train_with_cell_budget_widens_within_the_budget(tmp_path, monkeypatch):
    import tfmplayground.train as train_module
    from tfmplayground.train import WideningConfig

    monkeypatch.chdir(tmp_path)
    shapes = []
    real = train_module.add_mixed_widening_features

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        shapes.append(tuple(out.shape[1:]))
        return out

    monkeypatch.setattr(train_module, "add_mixed_widening_features", spy)
    torch.manual_seed(0)
    model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10)
    batches = []
    for _ in range(6):
        x, y = torch.randn(1, 80, 4), torch.randint(0, 3, (1, 80)).float()
        batches.append({"x": x, "y": y, "target_y": y.clone(), "train_test_split_index": 56})

    class Prior:
        num_steps = 6

        def __iter__(self):
            return iter(batches)

        def __len__(self):
            return 6

    config = WideningConfig(cell_budget=1200, width_min=20, width_max=60, min_rows=10, prob_no_widening=0.0,
                            include_original_prob=1.0)
    train(model, Prior(), nn.CrossEntropyLoss(), epochs=2, device=torch.device("cpu"), run_name="run",
          widening=config)
    assert len(shapes) == 12
    assert all(20 <= cols <= 60 and rows * cols <= 1200 + rows * 4 for rows, cols in shapes)  # + original columns
    assert len({s for s in shapes}) > 3  # shapes vary


def test_train_with_ramp_widens_nothing_in_the_first_epoch(tmp_path, monkeypatch):
    import tfmplayground.train as train_module
    from tfmplayground.train import WideningConfig

    monkeypatch.chdir(tmp_path)
    calls = []
    real = train_module.add_mixed_widening_features
    monkeypatch.setattr(train_module, "add_mixed_widening_features", lambda *a, **k: calls.append(1) or real(*a, **k))
    torch.manual_seed(0)
    model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10)
    x, y = torch.randn(1, 8, 3), torch.randint(0, 3, (1, 8)).float()
    batches = [{"x": x, "y": y, "target_y": y.clone(), "train_test_split_index": 5}] * 4

    class Prior:
        num_steps = 4

        def __iter__(self):
            return iter(batches)

        def __len__(self):
            return 4

    config = WideningConfig(add_features_min=5, add_features_max=5, prob_no_widening=0.0, ramp_epochs=1)
    train(model, Prior(), nn.CrossEntropyLoss(), epochs=2, device=torch.device("cpu"), run_name="run", widening=config)
    assert len(calls) == 4  # epoch 1: none; epoch 2 (ramp over): every batch


def test_cell_budget_widens_one_table_and_keeps_narrow_batches_whole(tmp_path, monkeypatch):
    """With several tables per batch, a widened batch keeps its first table (the budget is per table); narrow
    batches keep them all."""
    import tfmplayground.train as train_module
    from tfmplayground.train import WideningConfig

    monkeypatch.chdir(tmp_path)
    widened, seen = [], []
    real_widen, real_forward = train_module.add_mixed_widening_features, NanoTabPFNModel.forward
    monkeypatch.setattr(train_module, "add_mixed_widening_features",
                        lambda x, *a, **k: widened.append(x.shape[0]) or real_widen(x, *a, **k))

    def forward(self, *args, **kwargs):
        seen.append(args[0][0].shape[0])
        return real_forward(self, *args, **kwargs)

    monkeypatch.setattr(NanoTabPFNModel, "forward", forward)
    torch.manual_seed(0)
    model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10)
    x, y = torch.randn(3, 40, 4), torch.randint(0, 3, (3, 40)).float()
    batches = [{"x": x, "y": y, "target_y": y.clone(), "train_test_split_index": 28}] * 8

    class Prior:
        num_steps = 8

        def __iter__(self):
            return iter(batches)

        def __len__(self):
            return 8

    config = WideningConfig(cell_budget=800, width_min=10, width_max=20, min_rows=10, prob_no_widening=0.5)
    train(model, Prior(), nn.CrossEntropyLoss(), epochs=1, device=torch.device("cpu"), run_name="run",
          widening=config)
    assert widened and all(n == 1 for n in widened)  # widened batches: one table
    assert seen.count(3) == 8 - len(widened) and seen.count(1) == len(widened)  # narrow ones: all three
