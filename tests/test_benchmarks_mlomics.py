import numpy as np
import pandas as pd
import pytest

from tfmplayground.benchmarks import mlomics
from tfmplayground.benchmarks.metrics import accuracy, roc_auc
from tfmplayground.benchmarks.runner import compare, run_mlomics


def _fake_mlomics(root, dataset="BRCA", n_samples=60, n_genes=40, n_classes=3, seed=0):
    """Writes a tiny dataset with the MLOmics layout: features x samples CSVs and a label file."""
    rng = np.random.default_rng(seed)
    folder = root / "Main_Dataset" / "Classification_datasets" / f"GS-{dataset}" / "Original"
    folder.mkdir(parents=True)
    y = np.arange(n_samples) % n_classes
    values = rng.normal(size=(n_genes, n_samples)) + y[None, :] * (np.arange(n_genes) < 5)[:, None]
    genes = [f"g{i}" for i in range(n_genes)]
    df = pd.DataFrame(values, index=genes, columns=[f"s{j}" for j in range(n_samples)])
    df.iloc[3] = 0.0  # an all-zero gene, to be dropped
    df.iloc[7, 2] = np.nan  # a missing value, to be filled with the gene mean
    df = pd.concat([df, df.iloc[[10]]])  # a duplicated gene, to be dropped
    df.to_csv(folder / f"{dataset}_mRNA.csv")
    pd.DataFrame({"Label": y}).to_csv(folder / f"{dataset}_label_num.csv", index=False)
    return y


def test_load_cleans_like_tabpfn_wide(tmp_path):
    y_written = _fake_mlomics(tmp_path)
    X, y = mlomics.load("BRCA", tmp_path)
    assert X.shape == (60, 39)  # 40 genes - 1 all-zero; the duplicate is dropped
    assert "g3" not in X.columns and not X.isna().any().any()
    assert np.array_equal(y, y_written)


def test_reduce_features(tmp_path):
    _fake_mlomics(tmp_path)
    X, _ = mlomics.load("BRCA", tmp_path)
    assert mlomics.reduce_features(X, 0).shape == (60, 39)
    assert mlomics.reduce_features(X, 100).shape == (60, 39)  # more than available: unchanged
    assert mlomics.reduce_features(X, 10).shape == (60, 10)


def test_folds_are_deterministic_and_stratified():
    y = np.arange(100) % 4
    a, b = mlomics.folds(y), mlomics.folds(y)
    assert len(a) == 5
    for (tr1, te1), (tr2, te2) in zip(a, b):
        assert np.array_equal(te1, te2) and np.array_equal(tr1, tr2)
        assert np.bincount(y[te1]).tolist() == [5, 5, 5, 5]


def test_check_folds_match_detects_differences():
    y = np.arange(50) % 2
    rows = [
        {"dataset_name": "BRCA", "n_features": 30, "fold": i, "ground_truth": " ".join(map(str, y[te]))}
        for i, (_, te) in enumerate(mlomics.folds(y))
    ]
    published = pd.DataFrame(rows)
    assert mlomics.check_folds_match(published, "BRCA", 30, y)
    published.loc[0, "ground_truth"] = " ".join(["1"] * 10)
    assert not mlomics.check_folds_match(published, "BRCA", 30, y)


def test_metrics_match_tabpfn_wide_definition():
    y = np.array([0, 1, 1, 0])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.3, 0.7]])
    assert roc_auc(y, proba) == pytest.approx(0.75)
    assert accuracy(y, proba) == pytest.approx(0.75)
    y3 = np.array([0, 1, 2, 2])
    proba3 = np.eye(3)[y3] * 0.8 + 0.2 / 3
    assert roc_auc(y3, proba3) == pytest.approx(1.0)


def test_run_mlomics_with_random_forest_resumes(tmp_path):
    _fake_mlomics(tmp_path)
    out = tmp_path / "results.csv"
    res = run_mlomics(["random_forest"], tmp_path, out, datasets=("BRCA",), n_features=(0, 10), device="cpu")
    assert len(res) == 10  # 2 feature counts x 5 folds
    assert res.roc_auc.between(0, 1).all()
    again = run_mlomics(["random_forest"], tmp_path, out, datasets=("BRCA",), n_features=(0, 10), device="cpu")
    assert len(again) == 10  # nothing recomputed

    published = {"rf": res.assign(omic="mrna", checkpoint="x")}
    table = compare(res, published)
    assert set(table.checkpoint) == {"random_forest", "[TabPFN-Wide] rf"}
    assert (table["count"] == 5).all()


def test_external_model_names():
    from tfmplayground.benchmarks.models import is_checkpoint, parse_external

    assert parse_external("tabpfn-3.5") == ("tabpfn-3.5", 1)
    assert parse_external("tabpfn-3.5:n8") == ("tabpfn-3.5", 8)
    assert parse_external("tabpfn-3.5:auto") == ("tabpfn-3.5", "auto")
    assert not is_checkpoint("tabpfn-3.5:auto")
    assert not is_checkpoint("tabpfn-wide-5k") and not is_checkpoint("tabpfn-v2-gn2p4bpt:n8")
    assert not is_checkpoint("random_forest") and not is_checkpoint("logreg")
    assert is_checkpoint("workdir/graph_scm_base_w5000/epoch_100.pth")


def test_run_benchmark_baselines_resume_and_compare(tmp_path):
    from tfmplayground.benchmarks.analysis import mean_table, paired
    from tfmplayground.benchmarks.runner import run_benchmark

    _fake_mlomics(tmp_path / mlomics.FOLDER)
    out = tmp_path / "results.csv"
    models = ["random_forest", "logreg_en"]
    res = run_benchmark(["mlomics/BRCA/mrna"], models, out, tmp_path, n_features=(0, 10), device="cpu")
    assert len(res) == 2 * 2 * 5  # models x feature counts x folds
    assert set(res.n_features) == {39, 10} and res.roc_auc.between(0, 1).all() and (res.seconds >= 0).all()
    again = run_benchmark(["mlomics/BRCA/mrna"], models, out, tmp_path, n_features=(0, 10), device="cpu")
    assert len(again) == len(res)  # nothing recomputed

    assert list(mean_table(again).columns) == sorted(models)
    p = paired(again, "logreg_en", "random_forest")
    assert p["folds"] == 10 and p["wins"] + p["ties"] + p["losses"] == 10


def test_paired_counts_and_sign():
    import pandas as pd

    from tfmplayground.benchmarks.analysis import paired

    base = {"benchmark": "b", "dataset": "d", "n_features": 5}
    rows = [{**base, "fold": f, "model": "a", "roc_auc": 0.9} for f in range(4)]
    rows += [{**base, "fold": f, "model": "ref", "roc_auc": v} for f, v in enumerate([0.8, 0.85, 0.9, 0.95])]
    p = paired(pd.DataFrame(rows), "a", "ref")
    assert (p["folds"], p["wins"], p["ties"], p["losses"]) == (4, 2, 1, 1)
    assert p["mean_diff"] == pytest.approx(0.025)


def test_xgboost_and_categorical_columns():
    pytest.importorskip("xgboost")
    from tfmplayground.benchmarks.models import make_model, predict_proba

    rng = np.random.default_rng(0)
    X = np.column_stack([rng.integers(0, 3, 80), rng.normal(size=(80, 4))]).astype(np.float32)
    y = (X[:, 0] == 1).astype(int)
    for name in ("xgboost", "logreg_en"):
        proba = predict_proba(make_model(name, "cpu", categorical=[0]), X[:60], y[:60], X[60:])
        assert proba.shape == (20, 2) and roc_auc(y[60:], proba) > 0.9


def test_low_memory_label_only_for_our_checkpoints(tmp_path, monkeypatch):
    from tfmplayground.benchmarks import runner

    _fake_mlomics(tmp_path / mlomics.FOLDER)
    seen = []
    real = runner.make_model

    def spy(name, device, amp_dtype=None, categorical=None, low_memory=False):
        seen.append((name, low_memory))
        return real("random_forest", "cpu")

    monkeypatch.setattr(runner, "make_model", spy)
    res = runner.run_benchmark(["mlomics/BRCA/mrna"], ["ckpt.pth", "random_forest"], tmp_path / "r.csv", tmp_path,
                               device="cpu", low_memory=True)
    assert set(res.model) == {"ckpt.pth [low-memory]", "random_forest"}
    assert ("ckpt.pth", True) in seen and ("random_forest", False) in seen


def test_paired_leaves_out_nan_folds():
    import pandas as pd

    from tfmplayground.benchmarks.analysis import paired

    base = {"benchmark": "b", "dataset": "d", "n_features": 5}
    rows = [{**base, "fold": f, "model": "a", "roc_auc": v} for f, v in enumerate([float("nan"), 0.9, 0.9])]
    rows += [{**base, "fold": f, "model": "ref", "roc_auc": v} for f, v in enumerate([float("nan"), 0.8, 0.95])]
    p = paired(pd.DataFrame(rows), "a", "ref")
    assert (p["folds"], p["wins"], p["losses"]) == (2, 1, 1)
    assert p["mean_diff"] == pytest.approx(0.025)
