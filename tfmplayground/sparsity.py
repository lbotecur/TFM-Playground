"""Weight-sparse training, after Gao et al. (2025), "Weight-sparse transformers have interpretable circuits"
(arXiv:2511.13653): after every optimizer step each weight matrix keeps only its largest-magnitude entries, so
that the trained model has few connections and its circuits are small enough to be read.

As in the paper: every matrix (and bias) keeps the same fraction of nonzeros; that fraction falls linearly from
1 (dense) to the target over the first part of training (anneal_fraction of the optimizer steps); gradients and
the optimizer's moments stay dense, so a pruned weight can come back if its gradient makes it large again; and
every output neuron keeps at least min_per_row inputs, so that none dies.

With schedule-free AdamW (our optimizer) the model's weights during training are y = beta1 * x + (1 - beta1) * z,
where x (the averaged weights, the ones saved and used for inference) and z (the base sequence) are what the
optimizer keeps. The mask is chosen by the magnitude of x and applied to x and z, so x, z and y share the same
zeros and the checkpoint is exactly as sparse as intended.

Matrices with a fused query/key/value projection (nn.MultiheadAttention's in_proj) are treated as three matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class WeightSparsity:
    """How sparse the weights become.

    Attributes:
        density: Fraction of nonzero entries kept in every weight matrix and bias at the end of the annealing
            (1 = dense, off).
        anneal_fraction: Share of the optimizer steps over which the density falls linearly from 1 to density.
        min_per_row: Fewest nonzero inputs kept for every output neuron (all of them if it has fewer).
    """

    density: float = 1.0
    anneal_fraction: float = 0.5
    min_per_row: int = 4


def sparse_parameters(model: nn.Module) -> list[tuple[str, torch.nn.Parameter, int]]:
    """The parameters made sparse, with the number of parts each is split into (3 for a fused q/k/v
    projection): the weights and biases of the linear layers and attention projections. Norm scales are left
    dense."""
    selected = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            selected.append((f"{module_name}.in_proj_weight", module.in_proj_weight, 3))
            if module.in_proj_bias is not None:
                selected.append((f"{module_name}.in_proj_bias", module.in_proj_bias, 3))
        elif isinstance(module, nn.Linear):  # includes attention out_proj (NonDynamicallyQuantizableLinear)
            selected.append((f"{module_name}.weight", module.weight, 1))
            if module.bias is not None:
                selected.append((f"{module_name}.bias", module.bias, 1))
    return selected


def topk_mask(values: torch.Tensor, density: float, min_per_row: int) -> torch.Tensor:
    """Boolean mask keeping the round(density x size) largest |values| (at least one), plus, for a matrix, the
    min_per_row largest of every row."""
    flat = values.abs().flatten()
    k = max(1, round(density * flat.numel()))
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[flat.topk(k).indices] = True
    mask = mask.view_as(values)
    if values.dim() == 2 and min_per_row > 0:
        per_row = min(min_per_row, values.shape[1])
        rows = values.abs().topk(per_row, dim=1).indices
        mask.scatter_(1, rows, True)
    return mask


class WeightSparsifier:
    """Applies the masks after every optimizer step (call step() right after optimizer.step()).

    Args:
        model: The model (unwrapped).
        optimizer: Its optimizer; schedule-free AdamW (state 'z') or any other.
        config: The target density and its schedule.
        total_steps: Optimizer steps of the whole training run (for the annealing schedule).
        start_step: Optimizer steps already done (when resuming).
    """

    def __init__(self, model: nn.Module, optimizer, config: WeightSparsity, total_steps: int, start_step: int = 0):
        self.parameters = sparse_parameters(model)
        self.optimizer = optimizer
        self.config = config
        self.total_steps = max(1, total_steps)
        self.steps = start_step

    def density(self, step: int | None = None) -> float:
        """Density at an optimizer step: 1 at step 0, falling linearly to the target at anneal_fraction."""
        step = self.steps if step is None else step
        anneal_steps = self.config.anneal_fraction * self.total_steps
        progress = 1.0 if anneal_steps <= 0 else min(1.0, step / anneal_steps)
        return 1.0 - (1.0 - self.config.density) * progress

    @torch.no_grad()
    def step(self) -> None:
        """Counts an optimizer step and zeroes, in every sparse parameter, all but its largest entries."""
        self.steps += 1
        density = self.density()
        if density >= 1.0:
            return
        beta1 = None
        for group in self.optimizer.param_groups:
            if "betas" in group and group.get("train_mode", False):
                beta1 = group["betas"][0]
        for _, parameter, parts in self.parameters:
            state = self.optimizer.state.get(parameter, {})
            z = state.get("z")
            # schedule-free in train mode: parameter = y; x = (y - (1 - beta1) z) / beta1 is what is saved
            x = (parameter - (1.0 - beta1) * z) / beta1 if z is not None and beta1 is not None else parameter
            mask = torch.cat([topk_mask(chunk, density, self.config.min_per_row) for chunk in x.chunk(parts)])
            parameter.mul_(mask)
            if z is not None:
                z.mul_(mask.to(z.device))


@torch.no_grad()
def density_report(model: nn.Module) -> dict[str, float]:
    """Fraction of nonzero entries of every sparse parameter."""
    return {name: (parameter != 0).float().mean().item() for name, parameter, _ in sparse_parameters(model)}
