# Prior audit: what graph_scm gives the model, and what inference gives it

Written on 2026-10-13, from the measurements of 2026-10-12 and `experiments/prior_audit.py`. Context: on
OpenML-CC18 (narrow tables) v2 fails on tic-tac-toe, car and dna, so we checked whether the tables the model is
trained on have the form of the tables it gets at inference. No model was trained for this.

## The two paths into the model

| Step | Prior (training, graph_scm) | Inference (NanoTabPFNClassifier) |
|---|---|---|
| Categorical columns | codes 0..K-1, order arbitrary (random cluster centres) | codes 0..K-1 from an OrdinalEncoder, order arbitrary (alphabetical) |
| Clamp of outliers | **yes: about 4 robust std, over train and test rows** (`outlier_removing`) | **no** (only the model's clip at ±10) |
| z-score before the model | yes, over train and test rows (`standard_scaling`) | no |
| Constant columns | dropped (over all rows) | dropped (over the train rows) |
| Column order | random | numeric columns first, categorical ones last |
| Missing values | MCAR, injected by train() (up to 10%) | the real ones |
| Categorical mask | known by the prior, then lost; the widening guesses it (<= 20 values) | declared or inferred, used only to encode |
| Inside the model | z-score with the train rows, NaN -> 0 plus indicator, clip ±10 | the same |

The clamp and the z-score are in tabicl's `GraphSCM.__call__` (`tabicl/prior/_graph_scm.py`), not in our code.
The model never receives the column types: in both paths it sees numbers only.

## Measurements

`python experiments/prior_audit.py --tables 1280 --seed 0`: graph_scm with TabICL's defaults, 2-100 features,
40-1024 rows log-uniform as in training (1,280 tables, 57,871 columns). A first look on 2026-10-12 used tables of
1024 rows only; the numbers below are the ones with variable rows.

- **Categorical columns:** 38% of the columns. Cardinality is small: median 2, 99% have at most 9 categories
  (TabICL's default `use_corrected_cat_sizes=False` caps it below 10). Rare categories exist: in 22% of the
  categorical columns the rarest category is below 1/17 of the rows, in 4% below 1%.
- **The clamp does not erase rare categories.** A binary column whose rare value is beyond 4 std is clamped to
  {0, 4e-6} (the std floor is 1e-6), which the z-score turns back into the same binary column: 0 columns
  became constant. It merges a rare category into its neighbour in 1.3% of the categorical columns.
- **The clamp cuts the tails of the numeric columns:** 41% of them have a value changed, 0.8% of their values.
  This is the one real difference with inference, where heavy tails (counts, log-normal omics) reach the
  model up to the clip at ±10.
- **Raw scale:** before the clamp the numeric columns already have std <= 3.4 and |x| <= 23 (median of the
  per-table maximum 2.6); no infinite or missing values. Skipping TabICL's z-score is harmless numerically.
  (Tables that fail, with NaN, are turned by TabICL into labels -100, which our loader drops.)
- **What reaches the model** (|z| with the train rows, before the clip), share of the values above:

  | | > 4 | > 6 | > 10 | > 100 |
  |---|---|---|---|---|
  | graph_scm (TabICL) | 3.4e-3 | 8.1e-4 | 1.7e-4 | 4.7e-5 |
  | graph_scm_raw | 3.8e-3 | 1.5e-3 | 4.9e-4 | 7.7e-5 |

  Without the clamp the model sees about 3x more values beyond 10, the real tails, bounded by its clip.
  Above 100 are the columns constant on the train rows only (7% of the columns in tables of 40-1024 rows, 1.4%
  in tables of 1024 rows): the legacy ±100 clip was there for them.
- **Widening's categorical detection** (<= 20 values) misses no categorical column, and treats as categorical
  the 12% of the numeric columns that have <= 20 distinct values (mostly in small tables).
- **Reproducibility.** TabICL's graph generation depends on Python's string hashing, randomized per process: with
  the same seeds, two processes gave different tables; with `PYTHONHASHSEED` fixed they are identical. So the
  runs launched with `--seed` so far (the v2 included) were reproducible within a launch but not across
  launches. `pretrain.py` now restarts itself with `PYTHONHASHSEED` = the seed when `--seed` is given.

## Conclusions

1. The rare categories are not the reason for tic-tac-toe, car and dna. The likely reason is that their labels
   are pure logical rules (interactions without marginal effects), which are rare in graph_scm (H3 in
   `research_v2.md`); a later prior version can weight the interaction functions (`fct_types` with `prod`,
   `tree`) and allow more categories (`use_corrected_cat_sizes`).
2. To make the tables entering the model look like inference, `graph_scm_raw` (`external_priors/graph_raw.py`)
   gives the graph_scm tables without the clamp and the z-score (numeric columns raw, categorical ones as codes
   0..K-1) and keeps the categorical masks; the widening uses them instead of guessing and gives its new
   categorical columns codes 0..K-1 too. With `raw=False` the tables are TabICL's bit for bit, and the raw
   tables are TabICL's before its two steps (tests/test_graph_raw_prior.py), so the only change is the one
   intended: heavy tails reach the model as at inference, and no test rows are used to normalize.
3. Inference: putting the categorical columns last is not out of distribution (training columns are in random
   order and the column vectors do not encode the type), but it is a fixed pattern. `permute_columns=True` in
   NanoTabPFNClassifier/Regressor (or `--permute-columns` in eval_benchmark.py and validate_run.py) shuffles the
   preprocessed columns once with a fixed seed (`column_seed`, default 0), so a single forward pass stays
   deterministic and attention scores are mapped back to the original columns. It is off for checkpoints
   without the setting (the v2 and earlier, so the paper numbers do not change); `export_checkpoint(...,
   permute_columns=True)` stores it in a release, so the v3 models will use it by default.

Usage: `--prior-type graph_scm_raw` in `experiments/pretrain.py` (the graph_scm options apply). Not used by any
run yet; the next base model (long narrow training with variable rows, then widening, then a low-lr stage)
is meant to be trained with it.
