"""DataLoader and configuration for TabICL-based priors."""

import random

import numpy as np
import torch
from tabicl.prior import PriorDataset as TabICLPriorDataset
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def graph_prior_config(fct_types: str | None):
    """PriorConfig of graph_scm with the given random function types, or None for TabICL's defaults.
    fct_types is a comma-separated list of 'mlp', 'tree', 'disc', 'lin', 'quad', 'gp', 'em', 'prod' or the
    presets 'default' (all eight) and 'tabpfnv2'; a type listed twice is sampled twice as often, e.g.
    'default,tree,prod,quad' doubles the weight of trees, products and quadratic functions."""
    if not fct_types:
        return None
    from tabicl.prior.graph_lib._config import PriorConfig

    return PriorConfig(fct_types=fct_types)


def parse_prior_types(prior_type: str, weights: list[float] | None = None) -> tuple[list[str], list[float]]:
    """'graph_scm+tree_scm' -> (['graph_scm', 'tree_scm'], weights or equal weights): each batch comes from
    one of the priors, chosen with these probabilities."""
    types = prior_type.split("+")
    weights = list(weights) if weights else [1.0] * len(types)
    if len(weights) != len(types) or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError(f"prior weights {weights} do not match prior types {types}")
    return types, [w / sum(weights) for w in weights]


def _seed_generation(seed: int, epoch: int, worker: int):
    """Seeds the generators TabICL draws from (numpy's global one, torch, random) for one worker in one epoch,
    so with a run seed the tables of every epoch are fixed whatever happened before (e.g. a --resume)."""
    s = (seed * 1_000_003 + epoch * 10_007 + worker) % (2**32)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class _TabICLBatches(IterableDataset):
    """Yields usable batches (on the CPU) from TabICL priors. In a DataLoader with workers, every worker
    builds its own priors and yields its share of the steps, so the steps of an epoch add up to num_steps.
    Without a seed, each worker gets a different random seed from the DataLoader. With one, each (epoch,
    worker) gets a fixed seed and the DataLoader returns the workers' batches in a fixed order, so the same
    seed and number of workers give the same tables."""

    def __init__(self, loader: "TabICLPriorDataLoader"):
        self.loader = loader

    def __iter__(self):
        worker = get_worker_info()
        workers, index = (worker.num_workers, worker.id) if worker else (1, 0)
        steps = self.loader.num_steps // workers + (index < self.loader.num_steps % workers)
        if worker is not None:
            torch.set_num_threads(1)  # one core per worker: the workers already use all the cores given
        seed = getattr(self.loader, "seed", None)
        if seed is not None:
            _seed_generation(seed, self.loader.epoch, index)
        for _ in range(steps):
            yield self.loader._next_valid_batch(to_device=False)


class TabICLPriorDataLoader(DataLoader):
    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        num_datapoints_min: int,
        num_datapoints_max: int,
        min_features: int,
        max_features: int,
        max_num_classes: int,
        device: torch.device,
        prior_type: str = "mix_scm",
        log_seq_len: bool = False,
        min_train_size: float = 0.1,
        max_train_size: float = 0.9,
        num_workers: int = 0,
        batch_size_per_gp: int | None = None,
        prior_weights: list[float] | None = None,
        graph_fct_types: str | None = None,
        seed: int | None = None,
        epoch: int = 0,
    ):
        """prior_type: one TabICL prior ('graph_scm', 'tree_scm', ...) or several joined by '+', each batch
        drawn from one of them with probabilities prior_weights (equal by default).
        num_workers: processes generating batches in parallel (0 = in this process, one batch at a time).
        Generating the tables on the CPU is what limits training speed, so give it the cores available.
        batch_size_per_gp: datasets per group of the batch; a group shares the prior's sampled
        hyperparameters. None = the whole batch is one group (what all runs before 2026-10 used).
        graph_fct_types: random function types of graph_scm (see graph_prior_config); None = defaults.
        seed: makes the tables reproducible: the same seed, number of workers and settings give the same tables
        in every epoch. None = random (all runs before 2026-10). epoch: epochs already done (for a resumed
        run, so it continues the sequence instead of repeating its first epochs)."""
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_datapoints_min = num_datapoints_min
        self.num_datapoints_max = num_datapoints_max
        self.min_features = min_features
        self.max_features = max_features
        self.max_num_classes = max_num_classes
        self.prior_type = prior_type
        self.device = device
        self.num_workers = num_workers
        self.prior_types, self.prior_weights = parse_prior_types(prior_type, prior_weights)
        self._kwargs = dict(
            batch_size=batch_size,
            batch_size_per_gp=batch_size_per_gp or batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_num_classes,
            min_seq_len=num_datapoints_min,
            max_seq_len=num_datapoints_max,
            log_seq_len=log_seq_len,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            n_jobs=1,
        )
        self.graph_fct_types = graph_fct_types
        self.seed = seed
        self.epoch = epoch  # incremented at the start of every epoch (iteration over the loader)
        self._priors = None  # built on first use, in the process (or worker) that generates
        self._batches = None
        if num_workers > 0:
            # With a seed the workers are started again every epoch, so each one gets that epoch's seed.
            self._batches = DataLoader(
                _TabICLBatches(self), batch_size=None, num_workers=num_workers,
                persistent_workers=seed is None, prefetch_factor=4,
            )

    @property
    def pd(self):
        """The TabICL prior(s) of this process: one PriorDataset per prior type."""
        if self._priors is None:
            self._priors = [
                TabICLPriorDataset(
                    prior_type=t, config=graph_prior_config(self.graph_fct_types) if t == "graph_scm" else None,
                    **self._kwargs,
                )
                for t in self.prior_types
            ]
        if not isinstance(self._priors, list):  # replaced by a test
            return self._priors
        return self._priors[0] if len(self._priors) == 1 else self._priors

    @pd.setter
    def pd(self, value):  # tests replace the generator with a fixed list of raw batches
        self._priors = value

    def _draw(self):
        priors = self.pd
        if not isinstance(priors, list):
            return next(priors)
        return next(random.choices(priors, weights=self.prior_weights)[0])

    def tabicl_to_ours(self, d, to_device: bool = True):
        x, y, active_features, seqlen, train_size = d
        # Datasets in a batch can keep different numbers of features after TabICL drops the constant
        # ones (the rest is zero padding). Keep up to the widest, so no dataset loses real features.
        active_features = int(active_features.max().item())
        x = x[:, :, :active_features]
        train_test_split_index = train_size[0].item()
        # graph_scm returns the class labels as int64 (mlp_scm/tree_scm as float); the model averages
        # the train labels (pad_targets), so hand float targets to train() for every prior type.
        y = y.float()
        device = self.device if to_device else torch.device("cpu")
        return dict(
            x=x.to(device),
            y=y.to(device),
            target_y=y.to(device),
            train_test_split_index=train_test_split_index,
        )

    def _next_valid_batch(self, to_device: bool = True):
        """Draws TabICL batches until one has at least one usable dataset, and drops the unusable
        ones: datasets left without features (TabICL removed them all as constant) or marked as
        failed (labels -100). Either would crash or corrupt training."""
        while True:
            x, y, active_features, seqlen, train_size = self._draw()
            keep = (active_features > 0) & (y.reshape(y.shape[0], -1).min(dim=1).values >= 0)
            if keep.any():
                return self.tabicl_to_ours(
                    (x[keep], y[keep], active_features[keep], seqlen[keep], train_size[keep]), to_device
                )

    def __iter__(self):
        if getattr(self, "seed", None) is not None:
            self.epoch += 1
        if getattr(self, "_batches", None) is not None:
            return iter(self._batches)
        return iter(_TabICLBatches(self))

    def __len__(self):
        return self.num_steps
