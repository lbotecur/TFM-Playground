from __future__ import annotations

import torch


def inject_mcar_missingness(
    x: torch.Tensor,
    rate: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sets feature cells to NaN Missing-Completely-At-Random (MCAR).

    Each cell of ``x`` is independently set to NaN with probability ``rate``. This is a
    training-time augmentation: with the selective NaN filter in place, these NaNs reach
    the model, which imputes them and flags them via the indicator channel, so the indicator
    weights receive gradient and learn. Apply it to FEATURES only, never to targets.

    Args:
        x: feature tensor (any shape).
        rate: per-cell probability of being set to NaN, in [0, 1].
        generator: optional torch.Generator for reproducibility.

    Returns:
        A new tensor (the input is never modified). ``rate == 0`` returns an unchanged copy.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}.")
    x = x.clone()
    if rate == 0.0:
        return x
    mask = torch.rand(x.shape, generator=generator, device=x.device) < rate
    x[mask] = float("nan")
    return x


def add_widening_features(
    x: torch.Tensor,
    num_features_to_add: int,
    sparsity: float,
    noise_std: float,
    include_original_prob: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Continuous feature widening (Algorithm 1 of TabPFN-Wide).

    Generates ``num_features_to_add`` new features as a sparse linear projection of the
    original ones plus feature-dependent Gaussian noise, mimicking the strong inter-feature
    correlations of HDLSS data. With probability ``include_original_prob`` the original
    features are appended and the whole feature order is permuted.

    Operates on the last dimension (features); all leading dimensions are treated as batch.
    Per the paper, the noise std of each new feature is proportional to that feature's own
    std over the SAMPLES dimension (the second-to-last axis).

    Args:
        x: feature tensor of shape (..., num_samples, num_features).
        num_features_to_add: how many new features to generate (>= 0).
        sparsity: probability that a projection weight is kept (Bernoulli(p)); paper uses [0, 0.05].
        noise_std: noise scale sigma; paper uses [0, 1].
        include_original_prob: probability of appending the originals and permuting.
        generator: optional torch.Generator for reproducibility.

    Returns:
        A new tensor (the input is never modified).
    """
    if num_features_to_add < 0:
        raise ValueError(f"num_features_to_add must be >= 0, got {num_features_to_add}.")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}.")
    if noise_std < 0:
        raise ValueError(f"noise_std must be >= 0, got {noise_std}.")

    if num_features_to_add == 0:
        return x.clone()

    m = x.shape[-1]
    weights = torch.randn(m, num_features_to_add, generator=generator, device=x.device, dtype=x.dtype)
    mask = (torch.rand(m, num_features_to_add, generator=generator, device=x.device) < sparsity).to(x.dtype)
    x_wide = x @ (mask * weights)  # (..., num_features_to_add)

    # Feature-dependent noise: std of each new feature over the samples axis.
    stds = x_wide.std(dim=-2, keepdim=True)
    noise = torch.randn(x_wide.shape, generator=generator, device=x.device, dtype=x.dtype) * (noise_std * stds)
    x_wide = x_wide + noise

    if torch.rand(1, generator=generator, device=x.device).item() < include_original_prob:
        x_wide = torch.cat([x, x_wide], dim=-1)
        perm = torch.randperm(x_wide.shape[-1], generator=generator, device=x.device)
        x_wide = x_wide[..., perm]
    return x_wide

def detect_categorical_mask(x: torch.Tensor, max_unique: int = 20) -> torch.Tensor:
    """Boolean mask of categorical features (Algorithm 2 type detection, TabPFN-Wide).

    A feature is considered categorical if it has at most ``max_unique`` distinct
    observed (non-NaN) values, following the paper's rule (<= 20 distinct values);
    every other feature is treated as continuous. Distinct values are counted per
    feature over the SAMPLES axis, ignoring NaNs so that injected missingness never
    inflates a feature's cardinality.

    This is the WIDENING-time detection applied to the synthetic prior, and is
    deliberately separate from the inference-time categorical detection used on real
    data (which uses a different, smaller threshold calibrated for n ~ 150).

    Args:
        x: feature tensor of shape (num_samples, num_features).
        max_unique: maximum number of distinct values for a feature to count as categorical.

    Returns:
        Boolean tensor of shape (num_features,), True where the feature is categorical.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (num_samples, num_features), got shape {tuple(x.shape)}.")

    # Sort each column; torch places NaNs at the end. A distinct non-NaN value starts a
    # new group when it differs from the value above and neither cell is NaN.
    x_sorted, _ = torch.sort(x, dim=0)
    nan_sorted = torch.isnan(x_sorted)
    both_valid = (~nan_sorted[:-1]) & (~nan_sorted[1:])
    new_value = both_valid & (x_sorted[:-1] != x_sorted[1:])
    has_valid = (~nan_sorted).any(dim=0)
    n_unique = new_value.sum(dim=0) + has_valid.long()
    return n_unique <= max_unique


def split_widening_budget(cat_mask: torch.Tensor, num_features_to_add: int) -> tuple[int, int]:
    """Allocate the widening budget between continuous and categorical features.

    Following the paper: with categorical ratio ``r_cat = num_categorical / num_total``,
    allocate ``d_cat = floor(r_cat * num_features_to_add)`` categorical features and the
    remaining ``d_cont = num_features_to_add - d_cat`` continuous features. When there are
    no categorical features (r_cat = 0), the whole budget goes to continuous features (the
    all-continuous omics scenario).

    Args:
        cat_mask: boolean tensor of shape (num_features,) from ``detect_categorical_mask``.
        num_features_to_add: total number of new features to generate (d - m), >= 0.

    Returns:
        (num_continuous_to_add, num_categorical_to_add), summing to num_features_to_add.
    """
    if num_features_to_add < 0:
        raise ValueError(f"num_features_to_add must be >= 0, got {num_features_to_add}.")
    num_total = cat_mask.numel()
    if num_total == 0:
        raise ValueError("cat_mask must not be empty.")

    num_categorical = int(cat_mask.sum().item())
    r_cat = num_categorical / num_total
    num_cat_to_add = int(r_cat * num_features_to_add)  # floor for non-negative values
    num_cont_to_add = num_features_to_add - num_cat_to_add
    return num_cont_to_add, num_cat_to_add