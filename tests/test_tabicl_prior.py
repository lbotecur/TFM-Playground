import pytest
import torch


def test_tabicl_prior_loader_yields_batch():
    """Importing and iterating the TabICL loader must work (regression for the outdated
    tabicl.prior.dataset import path)."""
    pytest.importorskip("tabicl")
    from tfmplayground.external_priors import TabICLPriorDataLoader

    loader = TabICLPriorDataLoader(
        num_steps=1, batch_size=1,
        num_datapoints_min=50, num_datapoints_max=100,
        min_features=5, max_features=10,
        max_num_classes=3, device=torch.device("cpu"),
    )
    batch = next(iter(loader))
    assert "x" in batch and "y" in batch


def test_tabicl_prior_loader_respects_train_size_range():
    """min/max_train_size are forwarded to TabICL: the split stays within [0.3, 0.9] of the rows."""
    pytest.importorskip("tabicl")
    from tfmplayground.external_priors import TabICLPriorDataLoader

    loader = TabICLPriorDataLoader(
        num_steps=5, batch_size=1,
        num_datapoints_min=50, num_datapoints_max=100,
        min_features=5, max_features=10,
        max_num_classes=3, device=torch.device("cpu"),
        log_seq_len=True, min_train_size=0.3, max_train_size=0.9,
    )
    for batch in loader:
        n = batch["x"].shape[1]
        split = batch["train_test_split_index"]
        assert 0.3 * n - 1 <= split <= 0.9 * n + 1


def test_tabicl_to_ours_casts_integer_labels_to_float():
    """graph_scm returns class labels as int64; the loader must hand float targets to train()
    like the other prior types, because the model averages them (pad_targets)."""
    pytest.importorskip("tabicl")
    from tfmplayground.external_priors import TabICLPriorDataLoader

    loader = TabICLPriorDataLoader.__new__(TabICLPriorDataLoader)  # no prior generation needed
    loader.device = torch.device("cpu")
    x = torch.randn(2, 10, 5)
    y = torch.randint(0, 3, (2, 10))  # int64, as graph_scm returns
    d = (x, y, torch.tensor([5, 5]), torch.tensor([10, 10]), torch.tensor([6, 6]))

    out = loader.tabicl_to_ours(d)

    assert out["y"].dtype == torch.float32 and out["target_y"].dtype == torch.float32
    assert torch.equal(out["y"], y.float())


def _loader_with_batches(batches):
    """A TabICLPriorDataLoader whose TabICL generator is replaced by the given raw batches."""
    from tfmplayground.external_priors import TabICLPriorDataLoader

    loader = TabICLPriorDataLoader.__new__(TabICLPriorDataLoader)
    loader.device = torch.device("cpu")
    loader.num_steps = 1
    loader.pd = iter(batches)
    return loader


def test_loader_skips_batches_without_usable_datasets_and_drops_bad_ones():
    """A batch where every dataset has 0 features is skipped; in the next one, the dataset with
    0 features and the failed one (labels -100) are dropped and only the good one is returned."""
    pytest.importorskip("tabicl")
    x = torch.randn(3, 10, 6)
    y = torch.randint(0, 3, (3, 10)).float()
    y_failed = y.clone()
    y_failed[2] = -100.0
    all_empty = (x[:2], y[:2], torch.tensor([0, 0]), torch.tensor([10, 10]), torch.tensor([6, 6]))
    mixed = (x, y_failed, torch.tensor([0, 4, 5]), torch.tensor([10, 10, 10]), torch.tensor([6, 6, 6]))

    batch = next(iter(_loader_with_batches([all_empty, mixed])))

    assert batch["x"].shape == (1, 10, 4)  # only dataset 1 survives, with its 4 features
    assert torch.equal(batch["y"][0], y[1])


def test_loader_keeps_features_of_the_widest_dataset():
    """With different feature counts in a batch, no dataset is truncated: x keeps the widest."""
    pytest.importorskip("tabicl")
    x = torch.randn(2, 10, 8)
    y = torch.randint(0, 3, (2, 10)).float()
    batch = (x, y, torch.tensor([3, 7]), torch.tensor([10, 10]), torch.tensor([6, 6]))

    out = next(iter(_loader_with_batches([batch])))

    assert out["x"].shape == (2, 10, 7)


def _small_loader(**kwargs):
    from tfmplayground.external_priors import TabICLPriorDataLoader

    defaults = dict(num_steps=4, batch_size=2, num_datapoints_min=30, num_datapoints_max=40, min_features=3,
                    max_features=6, max_num_classes=3, device=torch.device("cpu"), prior_type="graph_scm")
    return TabICLPriorDataLoader(**{**defaults, **kwargs})


def test_loader_with_workers_yields_num_steps_different_batches():
    """With workers, an epoch still has exactly num_steps batches (split among the workers), on the CPU, and
    the workers do not repeat each other's tables (each is seeded differently)."""
    pytest.importorskip("tabicl")
    loader = _small_loader(num_steps=5, num_workers=2)
    for _ in range(2):  # persistent workers: a second epoch works too
        batches = list(loader)
        assert len(batches) == 5
        assert all(b["x"].device.type == "cpu" for b in batches)
    firsts = [b["x"][0, 0, 0].item() for b in batches]
    assert len(set(firsts)) == len(firsts)


def test_parse_prior_types_normalizes_weights_and_rejects_mismatch():
    from tfmplayground.external_priors.tabicl import parse_prior_types

    assert parse_prior_types("graph_scm") == (["graph_scm"], [1.0])
    assert parse_prior_types("graph_scm+tree_scm", [3, 1]) == (["graph_scm", "tree_scm"], [0.75, 0.25])
    with pytest.raises(ValueError):
        parse_prior_types("graph_scm+tree_scm", [1.0])


def test_mixture_of_priors_and_graph_function_types_generate():
    """A '+' mixture builds one TabICL prior per type, and graph_fct_types reaches graph_scm's PriorConfig."""
    pytest.importorskip("tabicl")
    from tfmplayground.external_priors.tabicl import graph_prior_config

    assert graph_prior_config(None) is None
    assert graph_prior_config("default,tree,prod").fct_types == "default,tree,prod"
    loader = _small_loader(prior_type="graph_scm+tree_scm", graph_fct_types="tree,prod", batch_size_per_gp=1)
    batches = list(loader)
    assert len(batches) == 4 and len(loader.pd) == 2
    assert loader.pd[0].prior.config.fct_types == "tree,prod"
    assert loader._kwargs["batch_size_per_gp"] == 1


def _firsts(loader, epochs=1):
    return [[b["x"][0, 0, 0].item() for b in loader] for _ in range(epochs)]


@pytest.mark.parametrize("workers", [0, 2])
def test_seeded_loader_gives_the_same_tables_and_new_ones_each_epoch(workers):
    """Same seed and workers: the same tables, epoch by epoch. Another seed: other tables. Each epoch differs
    from the previous one. A loader told that 1 epoch is done starts with the second epoch's tables (resume)."""
    pytest.importorskip("tabicl")
    a = _firsts(_small_loader(num_steps=4, num_workers=workers, seed=7), epochs=2)
    b = _firsts(_small_loader(num_steps=4, num_workers=workers, seed=7), epochs=2)
    other = _firsts(_small_loader(num_steps=4, num_workers=workers, seed=8))
    resumed = _firsts(_small_loader(num_steps=4, num_workers=workers, seed=7, epoch=1))
    assert a == b
    assert a[0] != a[1]
    assert other[0] != a[0]
    assert resumed[0] == a[1]
