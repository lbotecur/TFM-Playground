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

Graph_scm with TabICL's defaults, 2-100 features, up to 1024 rows (about 1,600 tables, 75,000 columns):

- **Categorical columns:** 42% of the columns. Cardinality is small: median 2, 99% have at most 9 categories
  (TabICL's default `use_corrected_cat_sizes=False` caps it below 10). Rare categories exist: in 15% of the
  binary columns the minority is below 1/17 of the rows, and the rarest category is below 0.1% in 1% of the
  categorical columns.
- **The clamp does not erase rare categories.** A binary column whose rare value is beyond 4 std is clamped to
  {0, 4e-6} (the std floor is 1e-6), which the z-score turns back into the same binary column. 0 columns
  became constant. It merges a rare category into its neighbour in 2.3% of the categorical columns.
- **The clamp cuts the tails of numeric columns:** 49% of them have a value changed, 0.6% of their values.
  This is the one real difference with inference, where heavy tails (counts, log-normal omics) reach the
  model up to the clip at ±10.
- **Raw scale:** before the clamp the numeric columns already have std <= 3.9 and |x| <= 32 (median of the
  per-table maximum 3.2); no infinite values; 1.6% of the tables have missing values, which TabICL turns into
  failed tables (labels -100) that our loader drops. Skipping TabICL's z-score is harmless numerically.
- **What reaches the model** (|z| with the train rows, before the clip): 0.30% of the values above 4, 0.075%
  above 6, 0.014% above 10, 0.001% above 100. Above 100 are the columns constant on the train rows only
  (1.4% of the columns): the legacy ±100 clip was there for them. The clip at ±10 changes 0.014% of the values.
- **Widening's categorical detection** (<= 20 values) misses no categorical column, and treats as categorical
  the 12% of the numeric columns that have <= 20 distinct values (mostly in small tables).

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
   order and the column vectors do not encode the type), but it is a fixed pattern; an optional fixed-seed
   column permutation is planned (off for v2, so the paper numbers do not change).

Usage: `--prior-type graph_scm_raw` in `experiments/pretrain.py` (the graph_scm options apply). Not used by any
run yet; the next base model (long narrow training with variable rows, then widening, then a low-lr stage)
is meant to be trained with it.
