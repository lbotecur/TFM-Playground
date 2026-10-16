"""Mixes tables of the mechanism omics prior (tfmplayground/priors/omics_mechanisms.py) into the batches of a base
prior, so that a training stage sees a share of omics tables and keeps the general ones (it never replaces them).

The base prior is iterated as always (same batches per epoch, same seeds, same resuming); each of its batches is
replaced, with probability `fraction`, by a batch of omics tables. Those are already wide, so they carry
`pre_widened=True` and train() does not widen them again (missing values are still added). With a seed, the choice
and the omics tables of every epoch are fixed, as the base prior's tables are.
"""

from __future__ import annotations

import numpy as np
import torch

from tfmplayground.priors.omics_mechanisms import MechanismPriorConfig, sample_dataset


class MixedOmicsPrior:
    """Batches of `base`, each replaced by omics tables with probability `fraction`.

    Args:
        base: The base prior (e.g. TabICLPriorDataLoader): iterable over one epoch's batches, with `len`.
        fraction: Share of the batches made of omics tables (0 = the base prior unchanged).
        config: Settings of the omics tables (rows, width...).
        batch_size: Omics tables per batch, padded with zero columns to the widest. Use 1 for wide tables.
        device: Where the batch tensors go.
        seed: Seed of the choices and of the omics tables (with the epoch); None = random.
        min_train_size, max_train_size: Range of the share of training rows of an omics batch.
    """

    def __init__(self, base, fraction: float, config: MechanismPriorConfig | None = None, batch_size: int = 1,
                 device: torch.device | str = "cpu", seed: int | None = None, min_train_size: float = 0.3,
                 max_train_size: float = 0.9):  # fmt: skip
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        self.base = base
        self.fraction = fraction
        self.config = config or MechanismPriorConfig()
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.min_train_size, self.max_train_size = min_train_size, max_train_size
        self.num_steps = len(base)
        self._epochs = 0  # epochs started, when the base prior does not count them

    @property
    def epoch(self) -> int:
        return getattr(self.base, "epoch", self._epochs)

    @epoch.setter
    def epoch(self, value: int) -> None:  # a resumed run continues the sequence (see pretrain.py)
        if hasattr(self.base, "epoch"):
            self.base.epoch = value
        self._epochs = value

    def __len__(self) -> int:
        return len(self.base)

    def __iter__(self):
        batches = iter(self.base)  # a seeded base prior counts its epoch here
        if not hasattr(self.base, "epoch"):
            self._epochs += 1
        rng = np.random.default_rng(None if self.seed is None else [self.seed, self.epoch])
        for batch in batches:
            yield self._omics_batch(rng) if rng.random() < self.fraction else batch

    def _omics_batch(self, rng) -> dict:
        c = self.config
        n_rows = int(round(np.exp(rng.uniform(np.log(c.min_rows), np.log(c.max_rows)))))
        tables = [sample_dataset(rng, c, n_rows=n_rows) for _ in range(self.batch_size)]
        x = np.zeros((len(tables), n_rows, max(t.X.shape[1] for t in tables)), dtype=np.float32)
        for i, t in enumerate(tables):
            x[i, :, : t.X.shape[1]] = t.X
        y = torch.from_numpy(np.stack([t.y for t in tables]).astype(np.float32)).to(self.device)
        # rows are i.i.d. and in random order: the first ones can be the training rows
        split = int(round(n_rows * rng.uniform(self.min_train_size, self.max_train_size)))
        return {"x": torch.from_numpy(x).to(self.device), "y": y, "target_y": y.clone(),
                "train_test_split_index": min(max(split, 1), n_rows - 1), "pre_widened": True}  # fmt: skip
