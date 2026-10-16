"""The mechanism omics prior (tfmplayground/priors/omics_mechanisms.py) keeps the rules of its specification:
mechanisms in layer order and disjoint, every variable of the label's mechanism with a path to the label, copies in
their source's layer, classes bounded by the rows, measurement on each layer's scale, and its ground truth
consistent with the table."""

import numpy as np
import pytest

from tfmplayground.priors.omics_mechanisms import (
    COPY,
    GENOMIC_LAYERS,
    INDEPENDENT,
    LABEL_MECHANISM,
    MIRNA_TARGETS,
    OTHER_MECHANISM,
    RANK,
    MechanismPriorConfig,
    sample_dataset,
)

SMALL = MechanismPriorConfig(max_total_features=200)  # narrow tables: fast, every rule still applies


@pytest.fixture(scope="module")
def tables():
    rng = np.random.default_rng(0)
    return [sample_dataset(rng, SMALL) for _ in range(150)]


def test_shapes_and_ground_truth_agree(tables):
    for d in tables:
        n, p = d.X.shape
        assert len(d.y) == n and d.y.min() >= 0 and d.y.max() < d.num_classes
        assert np.bincount(d.y, minlength=d.num_classes).min() >= 2
        for field in (d.layer, d.is_discrete, d.role, d.distance, d.mechanism, d.copy_kind):
            assert len(field) == p
        assert len(d.copy_sources) == p
        assert (d.role == LABEL_MECHANISM).sum() >= 2  # the label always has a mechanism of 2 or more variables
        assert set(d.layer) <= set(d.variant)


def test_edges_follow_the_layer_order_and_stay_inside_their_mechanism(tables):
    for d in tables:
        for parent, child, _ in d.edges:
            assert d.role[parent] in (LABEL_MECHANISM, OTHER_MECHANISM)
            if child == "label":
                assert d.role[parent] == LABEL_MECHANISM and d.distance[parent] == 1
                continue
            lp, lc = d.layer[parent], d.layer[child]
            assert RANK[lp] <= RANK[lc], (lp, lc)  # within a layer or forward, skipping absent layers
            assert lc not in GENOMIC_LAYERS  # genomic variables have no parents
            assert lp != "MIRNA" or lc in MIRNA_TARGETS
            assert d.mechanism[parent] == d.mechanism[child]  # mechanisms are disjoint


def test_every_variable_of_the_label_mechanism_reaches_the_label(tables):
    for d in tables:
        mech = np.flatnonzero(d.role == LABEL_MECHANISM)
        assert (d.distance[mech] >= 1).all()
        assert (d.distance[d.role != LABEL_MECHANISM] == -1).all()
        children = {}
        for parent, child, _ in d.edges:
            children.setdefault(parent, []).append(child)
        for node in mech:  # its distance is one more than its closest child's (the label counts as 0)
            dists = [0 if c == "label" else d.distance[c] for c in children.get(node, [])]
            assert dists and d.distance[node] == 1 + min(dists)


def test_independent_variables_have_no_edges(tables):
    for d in tables:
        in_edges = {p for p, _, _ in d.edges} | {c for _, c, _ in d.edges if c != "label"}
        assert not in_edges & set(np.flatnonzero(d.role == INDEPENDENT).tolist())
        assert (d.mechanism[d.role == INDEPENDENT] == -1).all()


def test_copies_come_from_graph_variables_of_their_layer(tables):
    for d in tables:
        for j in np.flatnonzero(d.role == COPY):
            sources = d.copy_sources[j]
            assert 1 <= len(sources) <= 3 and d.copy_kind[j] in ("linear", "non_linear")
            assert all(d.role[s] >= INDEPENDENT for s in sources)
            assert all(d.layer[s] == d.layer[j] for s in sources)
        assert all(not d.copy_sources[j] for j in np.flatnonzero(d.role != COPY))


def test_measurement_scales(tables):
    for d in tables:
        for layer in set(d.layer):
            x = d.X[:, d.layer == layer]
            if layer not in ("PROT", "METAB"):
                assert not np.isnan(x).any(), layer  # missing values only below a detection limit
            if layer == "METH":
                assert x.min() >= 0 and x.max() <= 1
            if layer in ("SNP", "MUT"):
                assert set(np.unique(x)) <= {0.0, 1.0, 2.0}
            if layer == "CNV" and d.variant["CNV"] == "discrete":
                assert set(np.unique(x)) <= {-2.0, -1.0, 0.0, 1.0, 2.0}


def test_classes_are_bounded_by_the_rows():
    rng = np.random.default_rng(1)
    assert {sample_dataset(rng, SMALL, n_rows=40).num_classes for _ in range(30)} == {2}  # 40 / 15 < 3
    many = [sample_dataset(rng, SMALL, n_rows=1000).num_classes for _ in range(80)]
    assert max(many) > 5 and np.mean(np.array(many) == 2) > 0.3


def test_layers_mostly_single_and_null_tables_rare(tables):
    single = np.mean([len(d.variant) == 1 for d in tables])
    assert 0.35 < single < 0.65
    assert np.mean([d.is_null for d in tables]) < 0.15


def test_graph_wider_than_the_maximum_width_is_not_widened():
    rng = np.random.default_rng(2)
    d = sample_dataset(rng, MechanismPriorConfig(max_total_features=5, num_independent=(30, 30)), n_rows=60)
    assert (d.role != COPY).all() and d.X.shape[1] >= 30


def test_same_seed_same_table():
    a = sample_dataset(np.random.default_rng(7), SMALL)
    b = sample_dataset(np.random.default_rng(7), SMALL)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)
    assert a.edges == b.edges
