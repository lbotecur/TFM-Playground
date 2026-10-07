# Training runs of the models in the paper

Reconstructed on 2026-10-07 from each run's `args.json` and launch command (the base run's `args.json` was
overwritten by a later `--resume`; its settings come from the original command and the per-epoch times in
its log). From now on every launch also writes `workdir/<run>/args_<start time>.json` with the command and
the git commit.

Architecture (all runs): `NanoTabPFNModel`, 12 layers, embedding 192, 6 heads, MLP 768, up to 10 classes,
7,288,138 parameters. Training: bf16 autocast, gradient checkpointing, 1000 steps per epoch, lr 1e-4 with
500 warmup steps, train fraction of each table 0.3-0.9, missing values up to 10%, a checkpoint every 5 epochs.
Every run uses the same effective batch of 8 tables per optimizer step (batch x gradient accumulation).

| | base | w5000 (main model) | w10000 | w5000_omics |
|---|---|---|---|---|
| Directory | `graph_scm_base` | `graph_scm_base_w5000` | `graph_scm_base_w10000` | `graph_scm_base_w5000_omics` |
| Checkpoint used | `epoch_100.pth` | `epoch_100.pth` | `epoch_100.pth` | `epoch_30.pth` |
| Starts from | scratch | base `epoch_100` | base `epoch_100` | w5000 `epoch_100` |
| Prior | graph_scm (TabICL) | graph_scm | graph_scm | omics |
| Features | 2-100 | 2-350 + widening | 2-350 + widening | 10-60 base variables, widened by the prior up to 5000 in total |
| Widening (added features) | none | 200-5000, none with prob. 0.7 | 200-10000, none with prob. 0.7 | none in train() (the prior widens) |
| Rows per table | 40-300 | 40-300 | 40-300 | 40-300 |
| Batch x accumulation | 8 x 1 | 2 x 4 | 1 x 8 | 2 x 4 |
| Epochs | 100 | 100 | 100 | 30 |

Widening settings fixed in `experiments/pretrain.py` (unchanged in its history): added features 200 to
`--add-features-max`, sparsity up to 0.05, noise up to 1.0, original features kept with prob. 0.5, up to 20
categories.

Launch commands:

    # base
    python -u experiments/pretrain.py --gpu 6 --run-name graph_scm_base --prior-type graph_scm \
        --add-features-max 0 --min-features 2 --max-features 100 --batch-size 8 --accumulate 1
    # w5000
    python -u experiments/pretrain.py --gpu 6 --run-name graph_scm_base_w5000 --prior-type graph_scm \
        --init-from workdir/graph_scm_base/epoch_100.pth --min-features 2 --max-features 350 \
        --add-features-max 5000 --prob-no-widening 0.7
    # w10000: as w5000 with --add-features-max 10000 --batch-size 1 --accumulate 8
    # w5000_omics
    python -u experiments/pretrain.py --run-name graph_scm_base_w5000_omics --prior-type omics \
        --init-from workdir/graph_scm_base_w5000/epoch_100.pth --max-features 5000 --epochs 30

Not used: the base run continued from `epoch_100` with `--batch-size 32 --accumulate 1` (`epoch_105` to
`epoch_145`, 2026-10). Against `epoch_100` on MLOmics mRNA and CNV (5 cancers each) it lost on 7 of 10
datasets in AUROC (mean -0.0025) and in accuracy (mean -0.020), so the base stays at `epoch_100`.
