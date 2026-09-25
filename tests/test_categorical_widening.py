import pytest
import torch

from tfmplayground.augmentation import (
    add_categorical_widening_features,
    add_mixed_widening_features,
    detect_categorical_mask,
    reduce_cardinality,
    sample_target_cardinalities,
    split_widening_budget,
)


def test_detect_categorical_threshold_boundary():
    """<= 20 distinct values is categorical; 21 is continuous; a constant is categorical."""
    col_20 = torch.arange(20.0).repeat(5)                       # 100 rows, 20 distinct
    col_21 = torch.cat([torch.arange(21.0), torch.zeros(79)])   # 100 rows, 21 distinct
    col_const = torch.zeros(100)                                # 100 rows, 1 distinct
    x = torch.stack([col_20, col_21, col_const], dim=1)
    mask = detect_categorical_mask(x, max_unique=20)
    assert mask.tolist() == [True, False, True]


def test_detect_categorical_ignores_nan():
    """NaNs are not counted as a category: cardinality is over observed values only."""
    col_20 = torch.arange(20.0).repeat(5)          # 100 rows, each of 20 values repeated 5x
    col_20[0] = float("nan")   # value 0.0 still present at indices 20,40,60,80 -> stays 20 distinct
    col_20[1] = float("nan")   # value 1.0 still present later -> stays 20 distinct
    col_21 = torch.arange(21.0).repeat(5)[:100]    # 100 rows, all 21 values present (each repeated)
    col_21[5] = float("nan")   # value 5.0 still present at indices 26,47,68,89 -> stays 21 distinct
    x = torch.stack([col_20, col_21], dim=1)
    mask = detect_categorical_mask(x, max_unique=20)
    assert mask.tolist() == [True, False]


def test_detect_categorical_single_row():
    """A single row means one observed value per column -> categorical."""
    x = torch.tensor([[3.14, -2.0, 0.0]])
    assert detect_categorical_mask(x).tolist() == [True, True, True]


def test_detect_categorical_rejects_non_2d():
    with pytest.raises(ValueError):
        detect_categorical_mask(torch.randn(2, 30, 5))


def test_split_budget_proportional():
    mask = torch.tensor([True, True, True] + [False] * 7)   # 3 categorical of 10
    assert split_widening_budget(mask, 100) == (70, 30)


def test_split_budget_all_continuous():
    mask = torch.zeros(10, dtype=torch.bool)                # omics scenario: no categoricals
    assert split_widening_budget(mask, 100) == (100, 0)


def test_split_budget_all_categorical():
    mask = torch.ones(10, dtype=torch.bool)
    assert split_widening_budget(mask, 50) == (0, 50)


def test_split_budget_uses_floor():
    mask = torch.tensor([True, False, False])              # r_cat = 1/3
    assert split_widening_budget(mask, 10) == (7, 3)       # floor(3.33) = 3


@pytest.mark.parametrize("num_cat,total,add", [(2, 7, 13), (5, 9, 101), (1, 4, 7), (8, 8, 3)])
def test_split_budget_always_sums_to_total(num_cat, total, add):
    mask = torch.tensor([True] * num_cat + [False] * (total - num_cat))
    d_cont, d_cat = split_widening_budget(mask, add)
    assert d_cont + d_cat == add
    assert d_cont >= 0 and d_cat >= 0


def test_split_budget_rejects_negative_add():
    with pytest.raises(ValueError):
        split_widening_budget(torch.ones(3, dtype=torch.bool), -1)


def test_split_budget_rejects_empty_mask():
    with pytest.raises(ValueError):
        split_widening_budget(torch.zeros(0, dtype=torch.bool), 10)


# ---- sample_target_cardinalities ----
def test_sample_target_cardinalities_range():
    g = torch.Generator().manual_seed(0)
    k = sample_target_cardinalities(5000, max_cats=20, generator=g)
    assert int(k.min()) >= 3 and int(k.max()) <= 20


def test_sample_target_cardinalities_biased_low():
    g = torch.Generator().manual_seed(0)
    k = sample_target_cardinalities(5000, max_cats=20, generator=g).float()
    assert k.mean().item() < 11.5   # below the midpoint of [3, 20]: exponential bias to low


def test_sample_target_cardinalities_reproducible():
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    assert torch.equal(
        sample_target_cardinalities(50, max_cats=15, generator=g1),
        sample_target_cardinalities(50, max_cats=15, generator=g2),
    )


def test_sample_target_cardinalities_empty():
    assert sample_target_cardinalities(0, max_cats=20).shape == (0,)


# ---- reduce_cardinality ----
def test_reduce_cardinality_enforces_bound():
    col = torch.arange(50.0).repeat_interleave(4)   # 50 distinct values
    g = torch.Generator().manual_seed(0)
    out = reduce_cardinality(col, max_cats=8, generator=g)
    assert out.unique().numel() <= 8
    assert out.shape == col.shape


def test_reduce_cardinality_keeps_original_values():
    col = torch.arange(50.0).repeat_interleave(4)
    g = torch.Generator().manual_seed(0)
    out = reduce_cardinality(col, max_cats=8, generator=g)
    assert set(out.unique().tolist()).issubset(set(col.unique().tolist()))


def test_reduce_cardinality_noop_when_already_small():
    col = torch.tensor([1.0, 1.0, 2.0, 3.0, 3.0, 3.0])   # 3 distinct
    assert torch.equal(reduce_cardinality(col, max_cats=10), col)


def test_reduce_cardinality_reproducible():
    col = torch.arange(40.0).repeat_interleave(5)
    g1 = torch.Generator().manual_seed(3)
    g2 = torch.Generator().manual_seed(3)
    assert torch.equal(reduce_cardinality(col, 6, generator=g1), reduce_cardinality(col, 6, generator=g2))


# ---- add_categorical_widening_features ----
def test_add_categorical_widening_shape():
    x_cat = torch.randint(0, 8, (150, 12)).float()
    g = torch.Generator().manual_seed(0)
    out = add_categorical_widening_features(x_cat, 200, sparsity=0.05, max_cats=20, generator=g)
    assert out.shape == (150, 200)


def test_add_categorical_widening_respects_max_cats():
    x_cat = torch.randint(0, 30, (200, 15)).float()   # donors with up to 30 categories
    g = torch.Generator().manual_seed(0)
    out = add_categorical_widening_features(x_cat, 100, sparsity=0.1, max_cats=12, generator=g)
    per_col_card = torch.tensor([out[:, j].unique().numel() for j in range(out.shape[1])])
    assert int(per_col_card.max()) <= 12


def test_add_categorical_widening_values_from_donors():
    x_cat = torch.randint(0, 8, (150, 10)).float()
    g = torch.Generator().manual_seed(0)
    out = add_categorical_widening_features(x_cat, 50, sparsity=0.3, max_cats=20, generator=g)  # k>1
    assert set(out.unique().tolist()).issubset(set(x_cat.unique().tolist()))


def test_add_categorical_widening_zero_features():
    x_cat = torch.randint(0, 8, (150, 10)).float()
    out = add_categorical_widening_features(x_cat, 0, sparsity=0.05, max_cats=20)
    assert out.shape == (150, 0)


def test_add_categorical_widening_does_not_mutate_input():
    x_cat = torch.randint(0, 8, (150, 10)).float()
    before = x_cat.clone()
    add_categorical_widening_features(x_cat, 30, sparsity=0.05, max_cats=20)
    assert torch.equal(x_cat, before)


def test_add_categorical_widening_reproducible():
    x_cat = torch.randint(0, 8, (150, 10)).float()
    g1 = torch.Generator().manual_seed(5)
    g2 = torch.Generator().manual_seed(5)
    a = add_categorical_widening_features(x_cat, 40, sparsity=0.05, max_cats=20, generator=g1)
    b = add_categorical_widening_features(x_cat, 40, sparsity=0.05, max_cats=20, generator=g2)
    assert torch.equal(a, b)


def test_add_categorical_widening_requires_donors():
    x_cat = torch.empty((150, 0))
    with pytest.raises(ValueError):
        add_categorical_widening_features(x_cat, 10, sparsity=0.05, max_cats=20)


# ---- add_mixed_widening_features (integration) ----
def test_mixed_widening_shape_without_originals():
    x = torch.randn(3, 50, 12)
    out = add_mixed_widening_features(x, 200, sparsity=0.05, noise_std=0.5, include_original_prob=0.0)
    assert out.shape == (3, 50, 200)


def test_mixed_widening_shape_with_originals():
    x = torch.randn(3, 50, 12)
    out = add_mixed_widening_features(x, 30, sparsity=0.05, noise_std=0.5, include_original_prob=1.0)
    assert out.shape == (3, 50, 42)   # 30 new + 12 originals


def test_mixed_widening_all_continuous():
    x = torch.randn(3, 50, 12)   # all high-cardinality -> all continuous (omics scenario)
    g = torch.Generator().manual_seed(0)
    out = add_mixed_widening_features(x, 100, sparsity=0.05, noise_std=0.5,
                                      include_original_prob=0.0, generator=g)
    assert out.shape == (3, 50, 100)


def test_mixed_widening_all_categorical_respects_max_cats():
    x = torch.randint(0, 6, (3, 50, 12)).float()   # all low-cardinality -> all categorical
    g = torch.Generator().manual_seed(0)
    out = add_mixed_widening_features(x, 100, sparsity=0.1, noise_std=0.5, max_cats=10,
                                      include_original_prob=0.0, generator=g)
    assert out.shape == (3, 50, 100)
    per_col_card = torch.tensor(
        [out[b, :, j].unique().numel() for b in range(out.shape[0]) for j in range(out.shape[2])]
    )
    assert int(per_col_card.max()) <= 10


def test_mixed_widening_mixed_input():
    x = torch.empty(3, 50, 12)
    x[:, :, :6] = torch.randint(0, 5, (3, 50, 6)).float()   # 6 categorical columns
    x[:, :, 6:] = torch.randn(3, 50, 6)                     # 6 continuous columns
    g = torch.Generator().manual_seed(0)
    out = add_mixed_widening_features(x, 200, sparsity=0.05, noise_std=0.5,
                                      include_original_prob=0.0, generator=g)
    assert out.shape == (3, 50, 200)


def test_mixed_widening_zero_features_returns_clone():
    x = torch.randn(3, 50, 12)
    out = add_mixed_widening_features(x, 0, sparsity=0.05, noise_std=0.5)
    assert out.shape == (3, 50, 12)
    assert torch.equal(out, x)


def test_mixed_widening_does_not_mutate_input():
    x = torch.randn(3, 50, 12)
    before = x.clone()
    add_mixed_widening_features(x, 30, sparsity=0.05, noise_std=0.5)
    assert torch.equal(x, before)


def test_mixed_widening_reproducible():
    x = torch.randn(3, 50, 12)
    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(11)
    a = add_mixed_widening_features(x, 40, sparsity=0.05, noise_std=0.5, generator=g1)
    b = add_mixed_widening_features(x, 40, sparsity=0.05, noise_std=0.5, generator=g2)
    assert torch.equal(a, b)


def test_mixed_widening_rejects_non_3d():
    with pytest.raises(ValueError):
        add_mixed_widening_features(torch.randn(50, 12), 10, sparsity=0.05, noise_std=0.5)