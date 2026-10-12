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
