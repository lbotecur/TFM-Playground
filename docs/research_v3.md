# Toward v3: a widening stage without the v2's blind spots

Fixed on 2026-10-12, before its runs. Not part of the first paper.

## What the v2 taught us

- Widening to 3000 columns (w3000r1024) made the model fragile on tables where almost every column is sparse
  (bag-of-words text: RELATHE, BASEHOCK, PCMAC), and the longer that stage trained the worse it got (RELATHE
  0.948 at epoch 30, 0.79 at 165, even with the clip). The same stage widened to 5000 columns did not show it
  at epoch 10 (0.942 against 0.914). The v2 needs a feature clip of ±10 at inference to hide it.
- 5000 columns x 1024 rows ran out of memory (80 GB); 5000 x 600 trained fine, but short tables alone cost
  madelon (0.79), which needs long tables (rows2048: 0.898 at e100).
- No dense prior table has more than half of its columns with |z| > 10 (at most 43% over 300 tables); text has
  95-98%.

## What changes (this PR adds the switches; all off by default, so earlier runs are unchanged)

1. `--widen-max-rows N`: widened batches keep N rows (the same share of training rows); narrow batches keep the
   prior's long tables up to `--max-rows`. Long-narrow and wide-short tables in the same run.
2. `--feature-clip 10` in training (already the default for new runs): the model is trained on what it sees at
   inference.
3. `--sparse-columns-prob p`: with probability p a widened batch gets sparse columns (a share up to
   `--sparse-columns-max-fraction` of them keep only their highest values, the rest 0; 0.2% to 20% nonzero per
   column), as text and single-cell data.
4. Validation: madelon, arcene and RELATHE (development datasets since 2026-10-11) are now part of
   `run_dev`/`validate_run.py`, and `validate_run.py --feature-clip 10` evaluates any run as the v2 is used.

## Branches (one change at a time, from rows2048 e100, the v2's starting point for widening)

| Branch | Widening | Rows | Clip in training | Sparse columns |
|---|---|---|---|---|
| v3a | up to 5000 | narrow up to 2048, widened 600 | 100 | no |
| v3b | as v3a | as v3a | 10 | no |
| v3c | as v3b | as v3b | 10 | p = 0.5 |

Same seed, workers and settings otherwise as the v2's widening stage. Each runs 30 epochs, as that stage, and is
validated every 5 epochs with `validate_run.py --feature-clip 10` (held-out prior loss, probes, development
datasets). Decision at epoch 30, against the v2's stage at its epoch 30 (w3000r1024 e30): RELATHE must stay
>= 0.93 at every snapshot (a branch that falls below is stopped), then the mean of madelon, arcene and the
HDLSS_dev datasets decides; differences under 0.005 are ties, broken toward the simpler branch. The winner then
gets the low learning-rate stage (3e-5, 30 epochs), as the v2.

## Results (2026-10-13)

AUROC on the development datasets (`validate_run.py --feature-clip 10`), epoch 30; reference: the v2's widening
stage at its epoch 30 (w3000r1024 e30). HDLSS_dev: GLIOMA, lymphoma, nci9, lung_discrete, warpPIE10P,
orlraws10P, USPS.

| Run | RELATHE | madelon | arcene | HDLSS_dev | Rule mean (madelon, arcene, HDLSS_dev) |
|---|---|---|---|---|---|
| reference | 0.870 | 0.876 | 0.948 | 0.973 | 0.932 |
| v3a | 0.946 | 0.844 | 0.944 | 0.976 | 0.921 |
| v3b | 0.933 | 0.814 | 0.946 | 0.975 | 0.912 |
| v3c | 0.952 | 0.866 | 0.946 | 0.969 | 0.927 |

- **By the rule, no branch wins.** None keeps RELATHE >= 0.93 at every snapshot (minimum 0.914, 0.872, 0.904 at
  single snapshots; the reference itself is at 0.870), and none beats the reference's mean. No low-lr stage.
- **RELATHE is fixed by every branch** (0.93-0.95 against 0.87 before any low-lr stage); v3c, with sparse
  columns, is the steadiest at the end (0.948-0.953 at epochs 20-30).
- **XOR probes improve a lot** (reference -> v3c: XOR-2 with 100 rows 0.895 -> 0.988, XOR-3 with 300 rows
  0.572 -> 0.662, with 1000 rows 0.585 -> 0.729): mixing long narrow tables with the wide ones keeps the
  ability to use interactions.
- **madelon loses** (0.81-0.87 against 0.876): widened tables have 600 rows at most, and madelon (500 columns,
  2600 rows) needs long tables of medium width. The rows of a widened table should depend on its width (a
  budget of rows x columns), not be one cap.
- **Clip 10 in training (v3b against v3a) shows no gain**, within the noise.
- **The validation is noisy:** madelon and USPS have 3 seeds and madelon moves by ±0.04 between snapshots of a
  run. Next time: more seeds, and the mean of the last snapshots instead of one.

These settle the widening stage of the next model (trained on `graph_scm_raw`): long narrow tables mixed with
wide ones, rows of widened tables by a rows x columns budget, sparse columns.
