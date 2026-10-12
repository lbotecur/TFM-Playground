"""Downloads the narrow subset of OpenML-CC18 (see tfmplayground/benchmarks/openml_cc18.py) to
<data_root>/OpenML-CC18/<name>.npz and writes index.csv. Needs the openml package and internet access
(pip install openml). Datasets already downloaded are skipped.

From the repo root:
    python experiments/download_openml_cc18.py
"""

import argparse
from pathlib import Path

import openml
import pandas as pd

from tfmplayground.benchmarks import openml_cc18 as cc18

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--data-root", default=str(REPO.parent))
args = parser.parse_args()

folder = Path(args.data_root) / cc18.FOLDER
folder.mkdir(parents=True, exist_ok=True)
rows = []
suite = openml.study.get_suite(cc18.SUITE_ID)
for task_id in suite.tasks:
    task = openml.tasks.get_task(task_id, download_splits=False)
    dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False, download_qualities=True)
    q = dataset.qualities
    n, d, c = int(q["NumberOfInstances"]), int(q["NumberOfFeatures"]) - 1, int(q["NumberOfClasses"])
    name = dataset.name
    if not cc18.keep(name, n, d, c):
        print(f"skip {name} ({n} rows, {d} features, {c} classes)", flush=True)
        continue
    path = folder / f"{name}.npz"
    if not path.exists():
        X, y, categorical, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
        X, y, cat_idx = cc18.to_arrays(X, y, categorical)
        cc18.save_task(path, X, y, cat_idx)
    X, y, cat_idx = cc18.load_task(name, args.data_root)
    rows.append({"name": name, "task_id": task_id, "dataset_id": task.dataset_id, "rows": X.shape[0],
                 "features": X.shape[1], "classes": int(y.max()) + 1, "categorical": len(cat_idx),
                 "missing": float(pd.isna(X).mean())})
    print(f"{name}: {X.shape[0]} rows, {X.shape[1]} features, {rows[-1]['classes']} classes, "
          f"{len(cat_idx)} categorical", flush=True)
pd.DataFrame(rows).to_csv(folder / "index.csv", index=False)
print(f"{len(rows)} datasets in {folder}")
