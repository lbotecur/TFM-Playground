"""Training on several processes (DistributedDataParallel, CPU/gloo here) gives the model that one process gives
with the same tables per optimizer step: 2 processes x accumulate a/2 = 1 process x accumulate a."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from tfmplayground.distributed import DistributedContext, split_batch
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.train import train


class _Prior:
    def __init__(self, batches):
        self._batches = batches
        self.num_steps = len(batches)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def _batches(n, nan_at=()):
    """n one-table batches, different tables; the training targets of the batches in nan_at are NaN."""
    generator = torch.Generator().manual_seed(0)
    batches = []
    for i in range(n):
        x = torch.randn(1, 10, 4, generator=generator)
        y = torch.randint(0, 3, (1, 10), generator=generator).float()
        if i in nan_at:
            y[0, 0] = float("nan")
        batches.append({"x": x, "y": y, "target_y": y.clone(), "train_test_split_index": 6})
    return batches


def _model(checkpointing):
    torch.manual_seed(0)
    model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10)
    model.gradient_checkpointing = checkpointing
    return model


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world_size, port, batches_per_rank, accumulate, checkpointing, workdir, epochs):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    os.chdir(workdir)
    context = DistributedContext(rank=rank, local_rank=rank, world_size=world_size)
    model, _ = train(_model(checkpointing), _Prior(batches_per_rank[rank]), nn.CrossEntropyLoss(), epochs=epochs,
                     accumulate_gradients=accumulate, device=torch.device("cpu"), run_name="ddp",
                     distributed=context)
    torch.save(model.state_dict(), os.path.join(workdir, f"rank{rank}.pt"))
    dist.destroy_process_group()


def _distributed(tmp_path, batches_per_rank, accumulate, checkpointing=False, epochs=1):
    world_size = len(batches_per_rank)
    mp.spawn(_worker, args=(world_size, _free_port(), batches_per_rank, accumulate, checkpointing, str(tmp_path),
                            epochs), nprocs=world_size, join=True)
    return [torch.load(tmp_path / f"rank{r}.pt") for r in range(world_size)]


def _single(tmp_path, batches, accumulate, checkpointing=False, epochs=1):
    os.chdir(tmp_path)
    model, _ = train(_model(checkpointing), _Prior(batches), nn.CrossEntropyLoss(), epochs=epochs,
                     accumulate_gradients=accumulate, device=torch.device("cpu"), run_name="single")
    return model.state_dict()


def _assert_same(a, b):
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=1e-4, atol=1e-6, msg=key)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_two_processes_train_the_model_of_one(tmp_path, monkeypatch, checkpointing):
    """1 process x accumulate 4 over b0..b7 = 2 processes x accumulate 2: each optimizer step averages the
    gradients of b0-b3, then b4-b7, whichever process computed them."""
    monkeypatch.chdir(tmp_path)
    b = _batches(8)
    (tmp_path / "s").mkdir(), (tmp_path / "d").mkdir()
    single = _single(tmp_path / "s", b, accumulate=4, checkpointing=checkpointing, epochs=2)
    per_rank = [[b[0], b[1], b[4], b[5]], [b[2], b[3], b[6], b[7]]]
    ranks = _distributed(tmp_path / "d", per_rank, accumulate=2, checkpointing=checkpointing, epochs=2)
    _assert_same(ranks[0], ranks[1])  # every process holds the same weights
    _assert_same(ranks[0], single)


def test_a_skipped_batch_is_skipped_by_every_process(tmp_path, monkeypatch):
    """A batch with NaN targets in one process makes all of them skip that position: no process waits forever
    for another, and their weights stay the same."""
    monkeypatch.chdir(tmp_path)
    b = _batches(6, nan_at={3})
    ranks = _distributed(tmp_path, [[b[0], b[2], b[4]], [b[1], b[3], b[5]]], accumulate=1)
    _assert_same(ranks[0], ranks[1])
    # the same as one process on the batches that both processes trained on: positions 0 and 2
    (tmp_path / "s").mkdir()
    single = _single(tmp_path / "s", [b[0], b[1], b[4], b[5]], accumulate=2)
    _assert_same(ranks[0], single)


@pytest.mark.parametrize(
    ("batch_size", "accumulate", "world_size", "expected"),
    [(1, 8, 4, (1, 2)), (1, 8, 8, (1, 1)), (8, 1, 4, (2, 1)), (8, 1, 8, (1, 1)), (2, 4, 8, (1, 1)), (4, 2, 4, (2, 1))],
)
def test_split_batch_keeps_the_tables_per_optimizer_step(batch_size, accumulate, world_size, expected):
    tables, steps = split_batch(batch_size, accumulate, world_size)
    assert (tables, steps) == expected
    assert tables * steps * world_size == batch_size * accumulate


def test_split_batch_refuses_uneven_splits():
    with pytest.raises(ValueError, match="cannot be split"):
        split_batch(1, 2, 4)
