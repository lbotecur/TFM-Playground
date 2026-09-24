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