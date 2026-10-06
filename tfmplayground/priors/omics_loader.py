"""Batches of the synthetic multi-omics prior in the format train() expects."""

import numpy as np
import torch

from tfmplayground.priors.omics import OmicsPriorConfig, sample_dataset


class OmicsPriorDataLoader:
    """Yields num_steps batches of omics datasets as dict(x, y, target_y, train_test_split_index).

    The model takes one split index per batch, so the datasets of a batch share the number of rows
    and the train/test split. Each dataset keeps its own width and is zero-padded to the widest, as
    in the TabICL loader. The prior already widens its datasets (config.max_total_features), so
    train() must not widen them again.
    """

    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        device: torch.device,
        config: OmicsPriorConfig | None = None,
        min_train_size: float = 0.3,
        max_train_size: float = 0.9,
        seed: int | None = None,
    ):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device
        self.config = config or OmicsPriorConfig()
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.rng = np.random.default_rng(seed)

    def _batch(self):
        c = self.config
        n_rows = int(round(np.exp(self.rng.uniform(np.log(c.min_rows), np.log(c.max_rows)))))
        datasets = [sample_dataset(self.rng, c, n_rows=n_rows) for _ in range(self.batch_size)]
        x = np.zeros((self.batch_size, n_rows, max(d.X.shape[1] for d in datasets)), dtype=np.float32)
        for i, d in enumerate(datasets):
            x[i, :, : d.X.shape[1]] = d.X
        y = torch.from_numpy(np.stack([d.y for d in datasets]).astype(np.float32)).to(self.device)
        # Rows are i.i.d. and in random order, so the first rows can be the training rows
        split = int(round(n_rows * self.rng.uniform(self.min_train_size, self.max_train_size)))
        return dict(
            x=torch.from_numpy(x).to(self.device),
            y=y,
            target_y=y,
            train_test_split_index=min(max(split, 1), n_rows - 1),
        )

    def __iter__(self):
        return iter(self._batch() for _ in range(self.num_steps))

    def __len__(self):
        return self.num_steps
