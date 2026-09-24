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