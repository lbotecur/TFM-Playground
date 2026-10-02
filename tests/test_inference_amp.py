"""Mixed-precision (bf16) inference: same autocast helper as training, close to fp32 results."""

import numpy as np
import pandas as pd
import torch

from tfmplayground.interface import NanoTabPFNClassifier
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import autocast

ARCH = dict(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=3)


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int) + (X[:, 2] > 1).astype(int)
    return X[:30], y[:30], X[30:]


def _classifiers():
    torch.manual_seed(0)
    model = NanoTabPFNModel(**ARCH).eval()
    fp32 = NanoTabPFNClassifier(model=model, device="cpu")
    bf16 = NanoTabPFNClassifier(model=model, device="cpu", amp_dtype=torch.bfloat16)
    return fp32, bf16


def test_autocast_helper_is_disabled_without_dtype():
    with autocast("cpu", None):
        assert not torch.is_autocast_enabled("cpu")
    with autocast("cpu", torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")


def test_bf16_predictions_are_float32_and_close_to_fp32():
    X_train, y_train, X_test = _data()
    fp32, bf16 = _classifiers()
    p32 = fp32.fit(X_train, y_train).predict_proba(X_test)
    p16 = bf16.fit(X_train, y_train).predict_proba(X_test)
    assert p16.dtype == np.float32
    assert np.allclose(p16.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(p16, p32, atol=0.05)


def test_bf16_attention_scores_and_embeddings_are_close_to_fp32():
    X_train, y_train, X_test = _data()
    fp32, bf16 = _classifiers()
    fp32.fit(X_train, y_train)
    bf16.fit(X_train, y_train)
    assert np.allclose(bf16.feature_attention_scores(X_test), fp32.feature_attention_scores(X_test), atol=0.02)
    e32, e16 = fp32.get_embeddings(X_test), bf16.get_embeddings(X_test)
    assert e16.dtype == np.float32
    assert np.allclose(e16, e32, atol=0.1)


def test_benchmark_runs_checkpoints_in_bf16_and_labels_the_rows(tmp_path):
    """make_model passes amp_dtype to our checkpoints (not to baselines), and run_mlomics labels the
    bf16 rows so that they never mix with fp32 results."""
    from tfmplayground.benchmarks.models import make_model
    from tfmplayground.benchmarks.runner import run_mlomics

    torch.manual_seed(0)
    ckpt = tmp_path / "tiny.pth"
    torch.save({"architecture": ARCH, "model": NanoTabPFNModel(**ARCH).state_dict()}, ckpt)
    assert make_model(str(ckpt), "cpu", torch.bfloat16).amp_dtype == torch.bfloat16

    folder = tmp_path / "Main_Dataset" / "Classification_datasets" / "GS-BRCA" / "Original"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    y = np.arange(40) % 2
    genes = pd.DataFrame(rng.normal(size=(8, 40)) + y, index=[f"g{i}" for i in range(8)])
    genes.columns = [f"s{j}" for j in range(40)]
    genes.to_csv(folder / "BRCA_mRNA.csv")
    pd.DataFrame({"Label": y}).to_csv(folder / "BRCA_label_num.csv", index=False)

    res = run_mlomics([str(ckpt)], tmp_path, tmp_path / "res.csv", datasets=("BRCA",), device="cpu",
                      amp_dtype=torch.bfloat16)
    assert len(res) == 5
    assert (res.checkpoint == f"{ckpt} [bfloat16]").all()
