import numpy as np
import pandas as pd

from tfmplayground.benchmarks import openml_cc18 as cc18


def _fake(root, name="toy", n=120):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "num": rng.normal(size=n),
        "cat": pd.Categorical(rng.choice(["a", "b", "c", None], size=n)),
        "txt": rng.choice(["x", "y"], size=n),  # object column: categorical even if not flagged
        "gap": np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n)),
    })
    y = rng.choice(["no", "yes", "maybe"], size=n)
    Xa, ya, cat = cc18.to_arrays(X, y, [False, True, False, False])
    (root / cc18.FOLDER).mkdir(parents=True, exist_ok=True)
    cc18.save_task(root / cc18.FOLDER / f"{name}.npz", Xa, ya, cat)
    pd.DataFrame([{"name": name}]).to_csv(root / cc18.FOLDER / "index.csv", index=False)
    return X, y


def test_to_arrays_codes_categoricals_and_keeps_missing_values(tmp_path):
    X, y = _fake(tmp_path)
    Xa, ya, cat = cc18.load_task("toy", tmp_path)
    assert Xa.dtype == np.float32 and Xa.shape == (120, 4) and cat == [1, 2]
    assert np.isnan(Xa[:, 1]).sum() == X["cat"].isna().sum()  # missing category -> NaN, not a code
    assert np.isnan(Xa[:, 3]).sum() == X["gap"].isna().sum()
    assert set(np.unique(ya)) == {0, 1, 2}
    assert cc18.datasets(tmp_path) == ["toy"]


def test_keep_applies_the_narrow_limits_and_leaves_madelon_out():
    assert cc18.keep("credit-g", 1000, 20, 2)
    assert not cc18.keep("madelon", 2600, 500, 2)
    assert not cc18.keep("big", 10_001, 20, 2)
    assert not cc18.keep("wide", 1000, 501, 2)
    assert not cc18.keep("many-classes", 1000, 20, 11)


def test_folds_are_stratified_10_fold_and_deterministic():
    y = np.arange(200) % 4
    a, b = cc18.folds(y), cc18.folds(y)
    assert len(a) == 10
    for (tr1, te1), (tr2, te2) in zip(a, b, strict=True):
        assert np.array_equal(te1, te2) and np.array_equal(tr1, tr2) and len(set(tr1) & set(te1)) == 0
        assert np.bincount(y[te1]).tolist() == [5, 5, 5, 5]


def test_runner_evaluates_an_openml_dataset(tmp_path):
    from tfmplayground.benchmarks.runner import run_benchmark

    _fake(tmp_path)
    res = run_benchmark(["openml/toy"], ["random_forest"], tmp_path / "r.csv", tmp_path, device="cpu")
    assert len(res) == 10 and set(res.benchmark) == {"openml"} and res.roc_auc.notna().all()
