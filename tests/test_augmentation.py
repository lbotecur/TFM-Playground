import pytest
import torch

from tfmplayground.augmentation import inject_mcar_missingness


def test_inject_mcar_rate_zero_is_noop():
    """rate=0 introduces no NaNs and returns an unchanged copy."""
    x = torch.randn(10, 5)
    out = inject_mcar_missingness(x, rate=0.0)
    assert not torch.isnan(out).any()
    assert torch.equal(out, x)


def test_inject_mcar_does_not_mutate_input():
    """The input tensor is never modified in place."""
    x = torch.randn(10, 5)
    x_before = x.clone()
    inject_mcar_missingness(x, rate=0.5)
    assert torch.equal(x, x_before)


def test_inject_mcar_matches_expected_rate():
    """Over many cells, the fraction of NaNs is close to the requested rate."""
    torch.manual_seed(0)
    x = torch.randn(200, 200)  # 40k cells
    out = inject_mcar_missingness(x, rate=0.3)
    frac = torch.isnan(out).float().mean().item()
    assert abs(frac - 0.3) < 0.02


def test_inject_mcar_is_reproducible_with_generator():
    """The same generator seed produces the same missingness mask."""
    x = torch.randn(50, 50)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    out1 = inject_mcar_missingness(x, rate=0.4, generator=g1)
    out2 = inject_mcar_missingness(x, rate=0.4, generator=g2)
    assert torch.equal(torch.isnan(out1), torch.isnan(out2))


def test_inject_mcar_rejects_invalid_rate():
    """A rate outside [0, 1] is an error."""
    x = torch.randn(4, 4)
    with pytest.raises(ValueError):
        inject_mcar_missingness(x, rate=1.5)
    with pytest.raises(ValueError):
        inject_mcar_missingness(x, rate=-0.1)