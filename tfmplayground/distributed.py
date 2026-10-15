"""Training on several GPUs with DistributedDataParallel: one process per GPU, launched with torchrun.

Each process generates its own tables (its own prior, with its own seed) and trains on them; after every
micro-batch that ends an accumulation group, DistributedDataParallel averages the gradients of all processes, so
every process takes the same optimizer step and keeps the same weights. With the batch split by split_batch, an
optimizer step sees as many tables as on one GPU (batch_size x accumulate), so a recipe trains the same model,
only faster: the same learning rate, warmup (in optimizer steps) and tables per epoch.

nn.DataParallel (pretrain.py --multi-gpu) splits one batch over the GPUs of one process: with a batch of one table
(the widening stages) there is nothing to split, all GPUs wait for the one prior, and it gave different results.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    """This process's place among the training processes."""

    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        """The process that logs and writes checkpoints."""
        return self.rank == 0


def init_distributed() -> DistributedContext | None:
    """Joins the process group when launched by torchrun (WORLD_SIZE > 1); None for a single process.

    NCCL on GPUs, gloo on the CPU. Each process uses the GPU of its LOCAL_RANK among the visible ones
    (choose them with CUDA_VISIBLE_DEVICES).
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return None
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size)


def split_batch(batch_size: int, accumulate: int, world_size: int) -> tuple[int, int]:
    """Batch and accumulation of each process, so that an optimizer step sees batch_size x accumulate tables.

    The accumulation is split first (1 x 8 on 4 GPUs: 1 x 2 each), then the batch (8 x 1 on 4 GPUs: 2 x 1 each).

    Returns:
        (tables per batch, batches per optimizer step) of each process.
    """
    if accumulate % world_size == 0:
        return batch_size, accumulate // world_size
    if world_size % accumulate == 0 and batch_size % (world_size // accumulate) == 0:
        return batch_size // (world_size // accumulate), 1
    raise ValueError(
        f"batch_size x accumulate = {batch_size} x {accumulate} cannot be split evenly over {world_size} processes"
    )


def any_process(flag: bool, device: torch.device) -> bool:
    """True if flag is True in any process (all of them must call it)."""
    value = torch.tensor([float(flag)], device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def from_main_process(flag: bool, device: torch.device) -> bool:
    """The main process's flag, in every process (all of them must call it)."""
    value = torch.tensor([float(flag)], device=device)
    dist.broadcast(value, src=0)
    return bool(value.item())


def mean_over_processes(value: float, device: torch.device) -> float:
    """The mean of a number over the processes (all of them must call it)."""
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    return tensor.item() / dist.get_world_size()
