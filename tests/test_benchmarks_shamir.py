import numpy as np
import pandas as pd

from tfmplayground.benchmarks import shamir


def _write_omic(path, samples, n_features, rng):
    """Shamir layout: a header of quoted sample names, then one whitespace-separated row per feature."""
    lines = [" ".join(f'"{s}"' for s in samples)]
    for j in range(n_features):
        lines.append(f'"f{j}" ' + " ".join(f"{v:.4f}" for v in rng.normal(size=len(samples))))
    path.write_text("\n".join(lines) + "\n")


def _fake_shamir(root):
    rng = np.random.default_rng(0)
    folder = root / shamir.FOLDER
    (folder / "breast").mkdir(parents=True)
    (folder / "clinical" / "clinical").mkdir(parents=True)
    samples = [f"TCGA.AA.{i:04d}.01" for i in range(30)]
    _write_omic(folder / "breast" / "exp", samples, 6, rng)
    _write_omic(folder / "breast" / "methy", samples[:-2], 4, rng)  # two patients without methylation
    _write_omic(folder / "breast" / "mirna", samples, 3, rng)
    labels = ["LumA", "Basal", "Her2"] * 10
    labels[5] = None  # a patient without subtype
    clinical = pd.DataFrame({"sampleID": [s.replace(".", "-") for s in samples], "PAM50Call_RNAseq": labels})
    clinical.to_csv(folder / "clinical" / "clinical" / "breast", sep="\t", index=False)
    return samples, labels


def test_load_aligns_omics_and_labels(tmp_path):
    samples, labels = _fake_shamir(tmp_path)
    X, y = shamir.load("breast", tmp_path, ("mrna", "methylation", "mirna"))
    kept = sorted(s.lower() for i, s in enumerate(samples[:-2]) if labels[i] is not None)
    assert list(X.index) == kept and X.shape == (27, 6 + 4 + 3)
    assert sorted(set(y)) == [0, 1, 2]
    X_task, y_task, categorical = shamir.load_task("breast/mrna", tmp_path)
    assert X_task.shape == (27, 6) and X_task.dtype == np.float32 and categorical == []
    assert np.array_equal(y_task, y)


def test_omics_are_standardized_over_all_samples(tmp_path):
    _fake_shamir(tmp_path)
    data = shamir._load_omic(tmp_path / shamir.FOLDER / "breast" / "exp")
    assert data.shape == (30, 6)
    np.testing.assert_allclose(data.mean().to_numpy(), 0, atol=1e-9)


def test_folds():
    y = np.arange(50) % 3
    assert len(shamir.folds(y)) == 5


def test_classes_with_fewer_patients_than_folds_are_left_out(tmp_path):
    samples, labels = _fake_shamir(tmp_path)
    clinical_path = tmp_path / shamir.FOLDER / "clinical" / "clinical" / "breast"
    clinical = pd.read_table(clinical_path)
    clinical.loc[[0, 1], "PAM50Call_RNAseq"] = "Normal"  # a class with only 2 patients
    clinical.to_csv(clinical_path, sep="\t", index=False)
    X, y = shamir.load("breast", tmp_path, ("mrna",))
    assert len(set(y)) == 3 and X.shape[0] == 25  # 27 aligned patients minus the 2 "Normal" ones
