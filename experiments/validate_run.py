"""Fixed validation of checkpoints (see tfmplayground/validation.py): held-out prior tables and synthetic
probes. This is what training choices are made on, never the paper benchmarks. Resumable: a checkpoint, set
and seed already in the output CSV is not evaluated again. With --watch it keeps waiting for the new
epoch_*.pth of the runs, so one process can follow several trainings on a spare GPU.

From the repo root, for example:
    # every snapshot of two runs, and keep following them (every 20 minutes)
    nohup python -u experiments/validate_run.py --runs workdir/v2_A workdir/v2_B --gpu 1 --watch 20 \\
        > workdir/validation/watch.log 2>&1 &
    # probes only, for external models (they have no prior loss)
    python experiments/validate_run.py --models tabpfn-3.5 tabpfn-wide-5k --skip-prior --gpu 1
    # how far from what the prior allows: ours and TabICL v2 on the same 50 held-out tables per set
    python experiments/validate_run.py --models workdir/graph_scm_base/epoch_100.pth tabicl-v2 --reference 50 \
        --skip-probes --gpu 1
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import torch

from tfmplayground.benchmarks.models import is_checkpoint
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.validation import (PRIOR_SETS, PROBES, load_prior_sets, prior_metrics, prior_metrics_classifier,
                                     probe_name, run_probes)

parser = argparse.ArgumentParser()
parser.add_argument("--runs", nargs="*", default=[], help="run directories: every epoch_*.pth in them")
parser.add_argument("--models", nargs="*", default=[], help="checkpoint paths or make_model names")
parser.add_argument("--every", type=int, default=5, help="with --runs, only epochs that are multiples of this")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--output", default="workdir/validation/results.csv")
parser.add_argument("--prior-dir", default="workdir/validation/prior_sets")
parser.add_argument("--prior-tables", type=int, default=200, help="held-out tables per prior set")
parser.add_argument("--seeds", type=int, default=5, help="seeds per probe")
parser.add_argument("--skip-prior", action="store_true")
parser.add_argument("--skip-probes", action="store_true")
parser.add_argument("--build-only", action="store_true", help="only build the held-out prior tables")
parser.add_argument("--reference", type=int, default=0,
                    help="also the held-out loss through each model's classifier interface on the first N tables "
                         "of each set, for ours and external models alike (e.g. tabicl-v2, the reference)")
parser.add_argument("--watch", type=float, default=0, help="minutes between looks for new checkpoints (0 = once)")
args = parser.parse_args()
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)

prior_sets = {} if args.skip_prior else load_prior_sets(args.prior_dir, args.prior_tables)
if args.build_only:
    raise SystemExit(f"Held-out prior tables in {args.prior_dir}: {', '.join(prior_sets)}")


def epoch_of(path: str) -> int | None:
    match = re.search(r"epoch_(\d+)\.pth$", path)
    return int(match.group(1)) if match else None


def candidates() -> list[str]:
    names = list(args.models)
    for run in args.runs:
        for path in sorted(Path(run).glob("epoch_*.pth"), key=lambda p: epoch_of(str(p))):
            if epoch_of(str(path)) % args.every == 0:
                names.append(str(path))
    return names


def done_keys() -> set:
    if not out.exists():
        return set()
    df = pd.read_csv(out)
    return set(zip(df.model, df.set, df.seed.astype(int)))


def save(rows: list[dict]):
    df = pd.DataFrame(rows)
    df.to_csv(out, mode="a", header=not out.exists(), index=False)


def evaluate(name: str, done: set):
    ours = is_checkpoint(name)
    base = {"model": name, "epoch": epoch_of(name)}
    if ours and prior_sets and any((name, s, 0) not in done for s in prior_sets):
        try:
            model = init_model_from_state_dict_file(name).to(device)
        except (RuntimeError, EOFError) as error:  # a snapshot still being written: next round
            print(f"skipping {name} for now: {error}", flush=True)
            return
        rows = []
        for set_name, tables in prior_sets.items():
            if (name, set_name, 0) in done:
                continue
            metrics = prior_metrics(model, tables, device)
            rows += [{**base, "kind": "prior", "set": set_name, "seed": 0, "metric": k, "value": v}
                     for k, v in metrics.items()]
        save(rows)
        del model
        torch.cuda.empty_cache()
    if args.reference and prior_sets:
        rows = []
        for set_name, tables in prior_sets.items():
            key = f"ref_{set_name}"
            if (name, key, 0) in done:
                continue
            metrics = prior_metrics_classifier(name, tables[: args.reference], device)
            rows += [{**base, "kind": "prior_ref", "set": key, "seed": 0, "metric": k, "value": v}
                     for k, v in metrics.items()]
        save(rows)
    if not args.skip_probes:
        todo = [p for p in PROBES if any((name, probe_name(*p), s) not in done for s in range(args.seeds))]
        if todo:
            save([{**base, "kind": "probe", **r} for r in run_probes(name, device, args.seeds, todo)])
    print(f"{time.strftime('%H:%M')} validated {name}", flush=True)


def summary():
    if not out.exists():
        return
    df = pd.read_csv(out)
    pd.set_option("display.width", 250)
    prior = df[(df.kind == "prior") & (df.metric == "loss")].pivot_table(index="model", columns="set", values="value")
    probes = df[df.kind == "probe"].pivot_table(index="model", columns="set", values="value")
    if len(prior):
        baseline = df[(df.kind == "prior") & (df.metric == "baseline_loss")].groupby("set").value.mean()
        prior.loc["(training-class frequencies)"] = baseline
        print("\nHeld-out prior loss (lower is better)\n")
        print(prior[[s for s in PRIOR_SETS if s in prior.columns]].round(4).to_string())
    ref = df[(df.kind == "prior_ref") & (df.metric == "loss")].pivot_table(index="model", columns="set", values="value")
    if len(ref):
        print("\nHeld-out prior loss through the classifier interface, first tables of each set (lower is better)\n")
        print(ref[[f"ref_{s}" for s in PRIOR_SETS if f"ref_{s}" in ref.columns]].round(4).to_string())
    if len(probes):
        print("\nProbe AUROC, mean over seeds\n")
        print(probes[[probe_name(*p) for p in PROBES if probe_name(*p) in probes.columns]].round(3).to_string())


while True:
    done = done_keys()
    for name in candidates():
        evaluate(name, done)
    summary()
    if not args.watch:
        break
    time.sleep(60 * args.watch)
