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
