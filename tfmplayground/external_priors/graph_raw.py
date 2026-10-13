"""graph_scm tables in the form our model gets at inference ("graph_scm_raw").

TabICL's GraphSCM builds a table as categorical columns (codes 0..K-1) followed by numeric columns, and then,
before handing it over: clamps every column to about 4 robust standard deviations (`outlier_removing`) and
z-scores it (`standard_scaling`), both over all the rows, train and test. At inference nothing like that
happens: numeric columns arrive as they are, categorical ones as codes 0..K-1, and the model normalizes with the
train rows only (normalize_features). This module gives the same tables without those two steps, so that what
enters the model in training has the form it has at inference; the model itself is unchanged. It also keeps
which columns are categorical (TabICL knows it but drops it), so the widening does not have to guess.

With raw=False the steps are applied as TabICL does, and the tables are bit for bit TabICL's (same seed); the
tests check this, so the only difference between the two priors is the one intended.
Measured on 2026-10-12 (docs/prior_audit.md): the raw numeric columns already have std <= 4 and |x| <= 32, with
no infinite values, so skipping the z-score is harmless numerically; the clamp touches 49% of the numeric
columns (0.6% of their values) and merges a rare category in 2.3% of the categorical ones.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tabicl.prior._dataset import GraphPrior, should_filter
from tabicl.prior._graph_scm import GraphSCM
from tabicl.prior._reg2cls import outlier_removing, standard_scaling
from tabicl.prior.graph_lib._base import Context, DatasetProperties
from tabicl.prior.graph_lib._dataset import RandomDataset
from tabicl.prior.graph_lib._properties import sample_categorical_sizes


class RawGraphSCM(GraphSCM):
    """GraphSCM that can skip TabICL's clamp and z-score (raw=True) and also returns the categorical mask.

    __call__ follows GraphSCM.__call__ (tabicl 2.2.0) line by line, with the same random draws in the same
    order, so raw=False reproduces it exactly. Returns X (seq_len, max_features), y (seq_len,) and a boolean
    mask (max_features,), True on the categorical columns (False on the padding)."""

    def __init__(self, *args, raw: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw = raw

    def __call__(self):
        context = Context(config=self.config, device=self.device)
        properties = DatasetProperties(
            n_train=self.seq_len,
            n_test=0,
            cat_sizes={
                "x": sample_categorical_sizes(self.num_features, context, max_cat_size=200),
                "y": [0 if self.regression else self.num_classes],
            },
        )
        data = RandomDataset(context).sample(properties).get_concat_tensors()
        X_cat, X_num = data.get("x_cat", None), data.get("x_num", None)
        parts = [part for part in (X_cat, X_num) if part is not None]  # categorical columns first
        if not parts:
            raise ValueError("No features found in dataset")
        X = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        num_categorical = 0 if X_cat is None else X_cat.shape[-1]

        X = X.float()
        if not self.raw:
            X = outlier_removing(X, threshold=4)
            X = standard_scaling(X)
        categorical = torch.arange(self.num_features, device=X.device) < num_categorical

        if self.permute_features:
            feat_perm = torch.randperm(self.num_features, device=self.device)
            X = X[..., feat_perm]
            categorical = categorical[feat_perm]

        if self.num_features < self.max_features:
            X = F.pad(X, (0, self.max_features - self.num_features), mode="constant", value=0.0)
            categorical = F.pad(categorical, (0, self.max_features - self.num_features), value=False)

        if self.regression:
            y = data["y_num"]
            if not self.raw:  # at inference regression targets are normalized by the training loop, not here
                y = standard_scaling(outlier_removing(y.float(), threshold=4))
            y = y.float().view(-1)
        else:
            y = data["y_cat"].view(-1)
            if self.permute_labels:
                class_perm = torch.randperm(self.num_classes, device=self.device)
                y = class_perm[y.long()]

        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):  # as TabICL: a failed table, dropped later
            X = torch.zeros_like(X)
            y = torch.full_like(y, -100.0)

        return X, y, categorical


def delete_constant_columns(X: torch.Tensor, d: torch.Tensor, categorical: torch.Tensor):
    """Prior.delete_unique_features for one table that also keeps the categorical mask aligned.
    X (1, T, H), d (1,), categorical (H,) -> the same with the constant columns removed (zero padding)."""
    xi, di = X[0], int(d[0])
    keep = torch.tensor([len(torch.unique(xi[:, j])) > 1 for j in range(di)], dtype=torch.bool)
    width, kept = xi.shape[-1], int(keep.sum())
    X_new = F.pad(xi[:, :di][:, keep], (0, width - kept), mode="constant", value=0)
    categorical_new = F.pad(categorical[:di][keep.to(categorical.device)], (0, width - kept), value=False)
    return X_new.unsqueeze(0), torch.tensor([kept], device=d.device), categorical_new


class RawGraphPrior(GraphPrior):
    """GraphPrior making its tables with RawGraphSCM. get_batch returns TabICL's five outputs plus the
    categorical masks (batch, max_features). Needs n_jobs=1 (the masks are collected in generation order;
    our DataLoader workers give the parallelism)."""

    def __init__(self, *args, raw: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw = raw
        self._masks: list[torch.Tensor] = []

    @torch.no_grad()
    def generate_dataset(self, params):
        # GraphPrior.generate_dataset (tabicl 2.2.0) with RawGraphSCM and the mask kept through the filters
        while True:
            X, y, categorical = RawGraphSCM(**params, raw=self.raw)()
            X, y = X.unsqueeze(0), y.unsqueeze(0)
            d = torch.tensor([params["num_features"]], device=self.device, dtype=torch.long)
            X, d, categorical = delete_constant_columns(X, d, categorical)
            if not (d > 0).all():
                continue
            if not (self.regression or self.cls_sanity_check(X, y, params["train_size"])):
                continue
            if should_filter(X[0], y[0], self.config, is_classif=not self.regression):
                continue
            self._masks.append(categorical)
            return X.squeeze(0), y.squeeze(0), d.squeeze(0)

    @torch.no_grad()
    def get_batch(self, batch_size=None):
        if self.n_jobs != 1:
            raise ValueError("RawGraphPrior needs n_jobs=1")
        if self.seq_len_per_gp:
            raise ValueError("RawGraphPrior does not support seq_len_per_gp")
        self._masks = []
        X, y, d, seq_lens, train_sizes = super().get_batch(batch_size)
        return X, y, d, seq_lens, train_sizes, torch.stack(self._masks).to(self.device)


class RawGraphPriorDataset:
    """Drop-in for tabicl's PriorDataset(prior_type="graph_scm") giving raw tables plus categorical masks."""

    def __init__(self, batch_size: int, raw: bool = True, config=None, **kwargs):
        self.batch_size = batch_size
        self.prior = RawGraphPrior(batch_size=batch_size, raw=raw, config=config, **kwargs)

    def get_batch(self, batch_size=None):
        return self.prior.get_batch(batch_size or self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()
