import pytest
import torch

from tfmplayground.augmentation import detect_categorical_mask, split_widening_budget


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