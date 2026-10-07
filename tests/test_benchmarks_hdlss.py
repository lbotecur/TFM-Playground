import numpy as np
import pandas as pd
import scipy.sparse
from scipy.io import savemat

from tfmplayground.benchmarks import hdlss


def _fake_mat(root, name, n=60, d=30, classes=(1, 2, 3), sparse=False):
    folder = root / hdlss.FOLDER
    folder.mkdir(exist_ok=True)
    y = np.array([classes[i % len(classes)] for i in range(n)])[:, None]
    X = np.random.default_rng(0).normal(size=(n, d))
    savemat(folder / f"{name}.mat", {"X": scipy.sparse.csc_matrix(X) if sparse else X, "Y": y})
    return X, y.ravel()


def test_load_task_shuffles_and_encodes_like_tabpfn_wide(tmp_path):
    from sklearn.utils import shuffle

    X_raw, y_raw = _fake_mat(tmp_path, "toy", classes=(1, 2, 3))
    X, y, categorical = hdlss.load_task("toy", tmp_path)
    X_ref, y_ref = shuffle(X_raw, y_raw, random_state=42)
    assert categorical == [] and X.dtype == np.float32 and X.shape == (60, 30)
    np.testing.assert_allclose(X, X_ref.astype(np.float32))
    assert np.array_equal(y, y_ref - 1)  # labels 1..3 -> 0..2


def test_sparse_mat_is_densified(tmp_path):
    _fake_mat(tmp_path, "sparse", sparse=True)
    X, _, _ = hdlss.load_task("sparse", tmp_path)
    assert isinstance(X, np.ndarray) and X.shape == (60, 30)


def test_folds_repeats_depend_on_size():
    assert len(hdlss.folds(np.arange(90) % 3)) == 30
    assert len(hdlss.folds(np.arange(2600) % 2)) == 9


def test_check_folds_match(tmp_path):
    y = np.arange(60) % 2
    rows = [{"dataset_name": "toy", "fold": i, "ground_truth": " ".join(map(str, y[te]))}
            for i, (_, te) in enumerate(hdlss.folds(y))]
    published = pd.DataFrame(rows)
    assert hdlss.check_folds_match(published, "toy", y)
    published.loc[3, "ground_truth"] = " ".join(["0"] * 20)
    assert not hdlss.check_folds_match(published, "toy", y)
