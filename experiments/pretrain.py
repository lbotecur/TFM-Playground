"""Preentrenamiento. Cada experimento es un comando; los valores por defecto son la línea base.

Ejemplos, desde la raíz del repo:
    nohup python -u experiments/pretrain.py --gpu 6 --run-name mix_scm --prior-type mix_scm > workdir/mix_scm.log 2>&1 &
    # continuar el mismo run si se cortó:
    nohup python -u experiments/pretrain.py --gpu 6 --run-name mix_scm --prior-type mix_scm --resume >> workdir/mix_scm.log 2>&1 &
    # run nuevo partiendo de los pesos de otro (continued pretraining):
    nohup python -u experiments/pretrain.py --gpu 6 --run-name mlp_scm_w8k --add-features-max 8000 \\
        --init-from workdir/mlp_scm/latest_checkpoint.pth > workdir/mlp_scm_w8k.log 2>&1 &
    # prior ómico (ya ensancha él: --max-features es su ancho total máximo y train() no ensancha):
    nohup python -u experiments/pretrain.py --gpu 6 --run-name omics --prior-type omics --max-features 5000 \\
        > workdir/omics.log 2>&1 &
    # on 4 GPUs (DistributedDataParallel, see tfmplayground/distributed.py): the same recipe, ~4x faster; the
    # GPUs are chosen with CUDA_VISIBLE_DEVICES (--gpu is ignored), --workers is per GPU
    CUDA_VISIBLE_DEVICES=0,2,3,5 nohup torchrun --standalone --nproc_per_node 4 experiments/pretrain.py \\
        --run-name mix_scm --prior-type mix_scm --workers 4 > workdir/mix_scm.log 2>&1 &
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"  # antes de importar torch

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.distributed import init_distributed, split_batch
from tfmplayground.external_priors import TabICLPriorDataLoader
from tfmplayground.interface import _migrate_state_dict
from tfmplayground.models.nanotabpfn import DEFAULT_FEATURE_CLIP, LEGACY_FEATURE_CLIP, NanoTabPFNModel
from tfmplayground.priors.omics import OmicsPriorConfig
from tfmplayground.priors.omics_loader import OmicsPriorDataLoader
from tfmplayground.priors.omics_mechanisms import MechanismPriorConfig
from tfmplayground.priors.omics_mechanisms_loader import MixedOmicsPrior
from tfmplayground.sparsity import WeightSparsity
from tfmplayground.train import WideningConfig, train

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", required=True)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--prior-type", default="mlp_scm")  # mlp_scm, mix_scm, tree_scm, graph_scm, graph_scm_raw, omics
parser.add_argument("--min-features", type=int, default=50)
parser.add_argument("--max-features", type=int, default=350)
parser.add_argument("--min-rows", type=int, default=40)
parser.add_argument("--max-rows", type=int, default=300)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--accumulate", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--add-features-max", type=int, default=5000)  # 0 desactiva el widening
parser.add_argument("--prob-no-widening", type=float, default=0.3)
parser.add_argument("--missing-rate-max", type=float, default=0.1)
parser.add_argument("--resume", action="store_true")  # continúa este run desde su latest_checkpoint.pth
parser.add_argument("--init-from", default=None)  # pesos iniciales de otro checkpoint (continued pretraining)
# Generation of the TabICL priors' tables (graph_scm, tree_scm, ...; not omics):
parser.add_argument("--workers", type=int, default=0,
                    help="processes generating tables in parallel; 0 = one, in the training process (runs before 2026-10)")
parser.add_argument("--batch-size-per-gp", type=int, default=None,
                    help="tables per group sharing the prior's hyperparameters; default = the whole batch")
parser.add_argument("--prior-weights", type=float, nargs="+", default=None,
                    help="with --prior-type a+b: probability of each prior per batch (default equal)")
parser.add_argument("--graph-fct-types", default=None,
                    help="graph_scm random function types, e.g. 'default,tree,prod,quad' (default: TabICL's)")
parser.add_argument("--seed", type=int, default=None,
                    help="seed of the model's initial weights and of the prior's tables (same seed, workers and "
                         "settings = same tables); default random, as all runs before 2026-10")
parser.add_argument("--filter-graphs", action="store_true",
                    help="graph_scm: resample graphs whose labels do not depend on any feature (as TabICLv2)")
parser.add_argument("--filter-datasets", action="store_true",
                    help="graph_scm: resample tables an ExtraTrees cannot predict better than a constant (as TabICLv2)")
parser.add_argument("--clip", type=float, default=1.0, help="maximum gradient norm (1.0 in all earlier runs)")
parser.add_argument("--widen-max-rows", type=int, default=0,
                    help="rows kept in widened batches (0 = all); narrow batches keep up to --max-rows")
parser.add_argument("--sparse-columns-prob", type=float, default=0.0,
                    help="probability that a widened batch gets sparse (zero-inflated) columns (0 in all runs before v3)")
parser.add_argument("--sparse-columns-max-fraction", type=float, default=1.0,
                    help="largest share of a widened batch's columns made sparse")
parser.add_argument("--tables-per-epoch", type=int, default=None,
                    help="tables per epoch, over all GPUs (an epoch is when a checkpoint is saved). Default 1000 x "
                         "--batch-size: 8000 for the base recipes (8 x 1), 1000 for the widening ones (1 x 8). Give "
                         "8000 with --batch-size 1 --accumulate 8 to count epochs as the base recipes do")
parser.add_argument("--cell-budget", type=int, default=0,
                    help="cells (rows x columns) of a widened table, 0 = off. When set, a widened table gets a width "
                         "drawn log-uniformly in [--widen-width-min, --widen-width-max] and at most "
                         "cell-budget / width rows (drawn log-uniformly from --min-rows): wider tables are shorter, "
                         "so every shape fits. "
                         "~4M cells fit on 80 GB (experiments/memory_budget.py). A widened batch keeps one table; "
                         "--add-features-max and --widen-max-rows are then not used")
parser.add_argument("--widen-width-min", type=int, default=200, help="with --cell-budget: narrowest widened table")
parser.add_argument("--widen-width-max", type=int, default=10000, help="with --cell-budget: widest widened table")
parser.add_argument("--widening-ramp-epochs", type=int, default=0,
                    help="epochs over which the share of widened batches rises from 0 to 1 - --prob-no-widening "
                         "(0 = from the start)")
# Architecture (defaults: the 7.3M-parameter model of every run so far) and weight-sparse training
# (tfmplayground/sparsity.py, after Gao et al. 2025, arXiv:2511.13653)
parser.add_argument("--embedding-size", type=int, default=192)
parser.add_argument("--num-heads", type=int, default=6)
parser.add_argument("--mlp-hidden-size", type=int, default=768)
parser.add_argument("--num-layers", type=int, default=12)
parser.add_argument("--norm", default="layer", choices=["layer", "rms"],
                    help="normalization of the blocks: layer (LayerNorm, all runs so far) or rms (RMSNorm, for "
                         "weight-sparse models: no mean or bias, so zeros stay zeros)")
parser.add_argument("--weight-density", type=float, default=1.0,
                    help="fraction of nonzero entries kept in every weight matrix at the end of the annealing "
                         "(1 = dense, off; e.g. 0.0625 keeps 1 in 16)")
parser.add_argument("--density-anneal-fraction", type=float, default=0.5,
                    help="share of the run over which the density falls linearly from 1 to --weight-density")
parser.add_argument("--min-nonzeros-per-row", type=int, default=4,
                    help="fewest nonzero inputs kept for every output neuron of a sparse matrix")
parser.add_argument("--mlp-topk", type=float, default=0.0,
                    help="fraction of the MLP hidden activations kept, the largest in magnitude (AbsTopK); 0 = all")
parser.add_argument("--omics-fraction", type=float, default=0.0,
                    help="share of the batches replaced by tables of the mechanism omics prior "
                         "(priors/omics_mechanisms.py), never widened again; 0 = off. Needs --batch-size 1")
parser.add_argument("--omics-max-features", type=int, default=3000,
                    help="widest omics table (its rows follow --min-rows/--max-rows)")
parser.add_argument("--feature-clip", type=float, default=None,
                    help="bound of the normalized features, saved in the checkpoint. Default: 10 for a new run (100 in "
                         "all runs before v2); a resumed run keeps the one it was trained with")
parser.add_argument("--multi-gpu", action="store_true",
                    help="old: split each batch over every visible GPU (nn.DataParallel); it gave different results. "
                         "Launch with torchrun instead (see the examples above)")
parser.add_argument("--no-log-seq-len", action="store_true",
                    help="rows per table uniform in [min-rows, max-rows] instead of log-uniform")
args = parser.parse_args()
# TabICL's graph generation depends on Python's string hashing (set and dict order), which is randomized per
# process unless PYTHONHASHSEED is fixed: with the same --seed, two launches gave different tables. With a seed,
# restart this process once with PYTHONHASHSEED set, so the tables are the same in every launch.
if args.seed is not None and os.environ.get("PYTHONHASHSEED") != str(args.seed):
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.execv(sys.executable, sys.orig_argv)  # orig_argv keeps the interpreter flags (-u)
if args.prior_type == "omics":
    args.add_features_max = 0  # el prior ómico ya ensancha; train() no vuelve a ensanchar

os.chdir(Path(__file__).resolve().parents[1])  # workdir/ siempre en la raíz del repo
# Launched by torchrun: one process per GPU, each on the GPU of its local rank (see tfmplayground/distributed.py)
distributed = init_distributed()
rank, world_size = (distributed.rank, distributed.world_size) if distributed else (0, 1)
is_main = rank == 0
if distributed is not None and args.multi_gpu:
    parser.error("--multi-gpu (nn.DataParallel) cannot be combined with torchrun")
device = torch.device(f"cuda:{distributed.local_rank if distributed else args.gpu}")
# Each process trains on its share of every optimizer step's tables: same tables per step and per epoch as one GPU
batch_size, accumulate = split_batch(args.batch_size, args.accumulate, world_size)
tables_per_epoch = args.tables_per_epoch or 1000 * args.batch_size
if tables_per_epoch % (world_size * batch_size * accumulate):
    step_tables = world_size * batch_size * accumulate
    parser.error(f"--tables-per-epoch {tables_per_epoch} must be a multiple of the {step_tables} tables of a step")
NUM_STEPS = tables_per_epoch // (world_size * batch_size)  # batches per epoch of each process
if args.cell_budget and batch_size * args.max_rows * args.max_features > args.cell_budget:
    # widened batches keep one table; narrow ones keep batch_size tables of up to max-rows x max-features cells
    parser.error(f"a narrow batch of {batch_size} tables x {args.max_rows} rows x {args.max_features} columns "
                 f"exceeds --cell-budget {args.cell_budget}: lower the tables per GPU or --max-rows")
if args.batch_size_per_gp and batch_size % args.batch_size_per_gp:
    parser.error(f"--batch-size-per-gp {args.batch_size_per_gp} must divide the {batch_size} tables per process")
# Each process draws its own tables: the prior's seed is the run's seed plus the rank (rank 0 = a one-GPU run)
prior_seed = None if args.seed is None else args.seed + rank
if world_size > 1 and is_main:
    print(f"{world_size} processes: {batch_size} tables x {accumulate} batches per optimizer step each, "
          f"{NUM_STEPS} batches per epoch each", flush=True)

# Guarda los parámetros del run junto a sus checkpoints
run_dir = Path("workdir") / args.run_name
run_dir.mkdir(parents=True, exist_ok=True)
# Each launch keeps its own record (command, git commit, start time), so a --resume with other options never
# overwrites the settings of earlier launches; args.json keeps those of the first launch.
started = datetime.now().strftime("%Y%m%d-%H%M%S")
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
record = {**vars(args), "command": " ".join(sys.argv), "git_commit": git("rev-parse", "HEAD"),
          "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")), "started": started}
record["world_size"] = world_size
if is_main:
    (run_dir / f"args_{started}.json").write_text(json.dumps(record, indent=2))
    if not (run_dir / "args.json").exists():
        (run_dir / "args.json").write_text(json.dumps(record, indent=2))

MAX_CLASSES = 10

if args.prior_type == "omics":
    # Ancho total log-uniforme hasta --max-features; --min-features no aplica (10-60 variables base)
    prior = OmicsPriorDataLoader(
        num_steps=NUM_STEPS,
        batch_size=batch_size,
        device=device,
        config=OmicsPriorConfig(
            min_rows=args.min_rows,
            max_rows=args.max_rows,
            max_total_features=args.max_features,
            max_classes=MAX_CLASSES,
        ),
        min_train_size=0.3,
        max_train_size=0.9,
    )
else:
    prior = TabICLPriorDataLoader(
        num_steps=NUM_STEPS,  # batches por época
        batch_size=batch_size,
        num_datapoints_min=args.min_rows,
        num_datapoints_max=args.max_rows,
        min_features=args.min_features,
        max_features=args.max_features,
        max_num_classes=MAX_CLASSES,
        device=device,
        prior_type=args.prior_type,
        log_seq_len=not args.no_log_seq_len,
        min_train_size=0.3,
        max_train_size=0.9,
        num_workers=args.workers,
        batch_size_per_gp=args.batch_size_per_gp,
        prior_weights=args.prior_weights,
        graph_fct_types=args.graph_fct_types,
        graph_filter_graphs=args.filter_graphs,
        graph_filter_datasets=args.filter_datasets,
        seed=prior_seed,
    )

if args.omics_fraction > 0:
    if args.prior_type == "omics":
        parser.error("--omics-fraction mixes omics tables into another prior, not into --prior-type omics")
    if distributed is not None:
        parser.error("--omics-fraction is for one-process runs (each process would draw its own omics batches)")
    if batch_size != 1:
        parser.error("--omics-fraction needs --batch-size 1: omics tables are up to --omics-max-features wide")
    prior = MixedOmicsPrior(
        prior, args.omics_fraction, MechanismPriorConfig(min_rows=args.min_rows, max_rows=args.max_rows,
                                                         max_total_features=args.omics_max_features),
        batch_size=batch_size, device=device, seed=None if prior_seed is None else prior_seed + 7919,
    )  # fmt: skip

if args.seed is not None:
    torch.manual_seed(args.seed)  # initial weights, and the widening and missing-value draws of train()
model = NanoTabPFNModel(
    embedding_size=args.embedding_size,
    num_attention_heads=args.num_heads,
    mlp_hidden_size=args.mlp_hidden_size,
    num_layers=args.num_layers,
    num_outputs=MAX_CLASSES,
    norm=args.norm,
    mlp_topk=args.mlp_topk,
).to(device)
model.gradient_checkpointing = True
if args.seed is not None and rank > 0:
    torch.manual_seed(args.seed + rank)  # same initial weights, but each process's own widening and missing draws

ckpt = None
if args.resume:
    ckpt = torch.load(run_dir / "latest_checkpoint.pth", map_location=device, weights_only=False)
    if hasattr(prior, "epoch"):
        prior.epoch = ckpt["epoch"]  # a seeded prior continues with the tables of the next epoch
    model.load_state_dict(ckpt["model"])
elif args.init_from:
    state = torch.load(args.init_from, map_location=device, weights_only=False)
    model.load_state_dict(_migrate_state_dict(state["model"]))

# Feature clip (saved in every checkpoint): the flag if given, else the resumed run's own (100 for runs started
# before v2, whose checkpoints do not record it), else the default for a new run, including --init-from (10).
if args.feature_clip is not None:
    feature_clip = args.feature_clip
elif ckpt is not None:
    feature_clip = ckpt["architecture"].get("feature_clip", LEGACY_FEATURE_CLIP)
else:
    feature_clip = DEFAULT_FEATURE_CLIP
model.feature_encoder.clip = float(feature_clip)
if is_main:
    print(f"feature clip: {feature_clip:g}", flush=True)

widening = WideningConfig(
    add_features_min=200,
    add_features_max=args.add_features_max,
    sparsity_max=0.05,
    noise_max=1.0,
    include_original_prob=0.5,
    max_cats=20,
    prob_no_widening=args.prob_no_widening,
    max_rows=args.widen_max_rows,
    sparse_columns_prob=args.sparse_columns_prob,
    sparse_columns_max_fraction=args.sparse_columns_max_fraction,
    cell_budget=args.cell_budget,
    width_min=args.widen_width_min,
    width_max=args.widen_width_max,
    min_rows=args.min_rows,
    ramp_epochs=args.widening_ramp_epochs,
)

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=nn.CrossEntropyLoss(),
    epochs=args.epochs,
    accumulate_gradients=accumulate,
    lr=args.lr,
    device=device,
    callbacks=[ConsoleLoggerCallback()],
    ckpt=ckpt,
    run_name=args.run_name,
    missing_rate_max=args.missing_rate_max,
    widening=widening,
    amp_dtype=torch.bfloat16,
    warmup_steps=500,
    snapshot_every=5,
    log_every=50,
    step_log_path=str(run_dir / "steps.csv"),
    grad_clip=args.clip,
    multi_gpu=args.multi_gpu,
    distributed=distributed,
    weight_sparsity=WeightSparsity(density=args.weight_density, anneal_fraction=args.density_anneal_fraction,
                                   min_per_row=args.min_nonzeros_per_row),
)
if is_main:
    print(f"Done. Checkpoint: workdir/{args.run_name}/latest_checkpoint.pth")
if distributed is not None:
    torch.distributed.destroy_process_group()
