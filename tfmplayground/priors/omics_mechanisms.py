"""Synthetic multi-omics tables built from mechanisms, with the mechanism of the label recorded as ground truth.

A label (a disease, a tumour subtype, a cell state) is produced by a molecular mechanism: a set of variables of one
or several omic layers and the relations between them. In real data the mechanism is hidden among thousands of
variables, other mechanisms, redundant copies and noise, and no ground truth exists to check a method against. Each
table generated here contains:

1. **One mechanism of the label**: a DAG of variables whose sink is the label; every variable has a path to it.
2. **Other mechanisms**: DAGs of the same kind that do not reach the label.
3. **Independent variables**: mechanisms of one variable.
4. **Widening**: linear and non-linear copies of those variables (always with less information than their
   source), pure noise and constant or near-constant columns.

Mechanisms are disjoint. Edges go within a layer or to any later layer in biological order (genome -> METH ->
MIRNA -> RNA -> PROT -> METAB), skipping absent layers; genomic variables have no parents. Relations are additive
(a sum of one function per parent: linear, saturating, threshold or non-monotone), so no effect appears only in
combination. Each variable's value in the graph is its true abundance; a measurement model of its layer (negative
binomial counts with library size for sequencing, beta values for methylation, left-censored intensities for
proteomics and metabolomics, ...) and the standard processing (log-CPM, ...) give the observed value.

Each table uses one or several omic layers, mostly one, and records per column its layer, its role and, for
copies, their sources. Design: the specification of 2026-10-16 (TabOmics v3 notes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LAYERS = ("SNP", "CNV", "MUT", "METH", "MIRNA", "RNA", "PROT", "METAB")
RANK = {layer: i for i, layer in enumerate(LAYERS)}
GENOMIC_LAYERS = ("SNP", "CNV", "MUT")  # fixed at birth: no parents
MIRNA_TARGETS = ("MIRNA", "RNA", "PROT")  # what a miRNA can regulate
# Probability of each layer being active in a table: expression first, as in real data and benchmarks
LAYER_WEIGHTS = {"RNA": 0.30, "METH": 0.15, "MIRNA": 0.15, "PROT": 0.10, "METAB": 0.10, "CNV": 0.08, "MUT": 0.07,
                 "SNP": 0.05}  # fmt: skip

# Role of each column (ground truth)
NOISE = 0  # pure noise column (widening)
CONSTANT = 1  # constant or near-constant column (widening)
COPY = 2  # linear or non-linear copy of graph variables, with less information (widening)
INDEPENDENT = 3  # a mechanism of one variable
OTHER_MECHANISM = 4  # in a mechanism that does not reach the label
LABEL_MECHANISM = 5  # in the mechanism of the label
ROLE_NAMES = {NOISE: "noise", CONSTANT: "constant", COPY: "copy", INDEPENDENT: "independent",
              OTHER_MECHANISM: "other_mechanism", LABEL_MECHANISM: "label_mechanism"}  # fmt: skip

FUNCTIONS = ("linear", "saturating", "threshold", "non_monotone")
FUNCTION_PROBS = (0.4, 0.3, 0.15, 0.15)


@dataclass
class MechanismPriorConfig:
    """Ranges from which every table samples its settings (the specification's values)."""

    min_rows: int = 40
    max_rows: int = 1024
    label_mechanism_size: tuple[int, int] = (2, 20)  # log-uniform
    num_other_mechanisms: tuple[int, int] = (0, 10)
    other_mechanism_size: tuple[int, int] = (2, 20)  # log-uniform
    num_independent: tuple[int, int] = (0, 30)
    max_parents: int = 3
    root_probability: float = 0.3  # chance that a node (not the first) has no parents
    node_snr: tuple[float, float] = (0.5, 5.0)  # variance of the signal / variance of the noise, log-uniform
    # Inverse temperature of the label (scale of its logits before Gumbel noise): sets the difficulty
    # log-uniform. Calibrated with experiments/omics_prior_diagnostics.py: with 1000 rows, 6% of the tables with a
    # signal gain < 0.05 accuracy from their mechanism (35% with (0.5, 8)); small tables stay hard, as small cohorts
    label_beta: tuple[float, float] = (2.0, 16.0)
    null_probability: float = 0.05  # tables whose label depends on nothing (recorded)
    min_rows_per_class: int = 15
    max_classes: int = 10
    max_total_features: int = 3000  # after widening; total width log-uniform up to this
    copy_linear_fraction: tuple[float, float] = (0.0, 0.5)
    copy_nonlinear_fraction: tuple[float, float] = (0.0, 0.3)
    noise_fraction: tuple[float, float] = (0.0, 0.5)
    constant_fraction: tuple[float, float] = (0.0, 0.05)
    copy_correlation: tuple[float, float] = (0.3, 0.95)  # correlation of a copy with its source signal (< 1)


@dataclass
class MechanismDataset:
    """A generated table and its ground truth. Rows are i.i.d. and in random order; columns are shuffled."""

    X: np.ndarray  # (rows, columns) float, NaN for missing values
    y: np.ndarray  # (rows,) int, classes 0..num_classes-1
    layer: np.ndarray  # (columns,) layer of each column
    is_discrete: np.ndarray  # (columns,) bool
    role: np.ndarray  # (columns,) role (NOISE ... LABEL_MECHANISM)
    distance: np.ndarray  # (columns,) for LABEL_MECHANISM: shortest path length to the label (1 = parent); else -1
    mechanism: np.ndarray  # (columns,) mechanism index (0 = the label's), -1 if none
    copy_sources: list  # (columns,) for COPY: column indices of the source variables in X; else []
    copy_kind: np.ndarray  # (columns,) "linear" / "non_linear" for COPY, "" otherwise
    num_classes: int
    is_null: bool  # True if the label depends on nothing
    edges: list  # (parent column, child column or "label", function) of every mechanism edge, in X's columns
    variant: dict  # per-layer measurement variant, e.g. {"RNA": "seq", "CNV": "discrete", "PROT": "ms"}


def _loguniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _loguniform_int(rng, low, high):
    return int(round(_loguniform(rng, max(low, 1), max(high, 1)))) if high >= 1 else 0


def _standardize(a):
    a = np.asarray(a, dtype=float)
    std = a.std(axis=0)
    return (a - a.mean(axis=0)) / np.where(std > 0, std, 1.0)


def _ranks(z):
    """Column-wise ranks in (0, 1)."""
    return (np.argsort(np.argsort(z, axis=0), axis=0) + 0.5) / z.shape[0]


# ---------------------------------------------------------------------------------------------------------
# Tables: rows, layers, classes


def _sample_layers(rng) -> list[str]:
    """Active layers: one with probability 0.5, two or three with 0.3, four or more with 0.2, by LAYER_WEIGHTS."""
    u = rng.random()
    count = 1 if u < 0.5 else int(rng.integers(2, 4)) if u < 0.8 else int(rng.integers(4, len(LAYERS) + 1))
    names = list(LAYER_WEIGHTS)
    p = np.array([LAYER_WEIGHTS[n] for n in names])
    chosen = rng.choice(names, count, replace=False, p=p / p.sum())
    return sorted(map(str, chosen), key=RANK.get)


def _sample_num_classes(rng, n_rows: int, config: MechanismPriorConfig) -> int:
    """2 (50%), 3-5 (35%), 6-10 (15%), at most rows / min_rows_per_class."""
    cap = max(2, min(config.max_classes, n_rows // config.min_rows_per_class))
    u = rng.random()
    k = 2 if u < 0.5 else int(rng.integers(3, 6)) if u < 0.85 else int(rng.integers(6, 11))
    return min(k, cap)


# ---------------------------------------------------------------------------------------------------------
# Mechanisms: DAGs in layer order


def _mechanism_layers(rng, size: int, active: list[str]) -> np.ndarray:
    """Layers of the nodes of one mechanism: one or several of the active layers, nodes ordered by layer."""
    n_layers = int(rng.integers(1, min(len(active), size) + 1))
    layers = sorted(rng.choice(active, n_layers, replace=False), key=RANK.get)
    counts = 1 + rng.multinomial(size - n_layers, np.full(n_layers, 1 / n_layers))
    return np.array([layer for layer, c in zip(layers, counts, strict=True) for _ in range(c)])


def _can_parent(parent_layer: str, child_layer: str) -> bool:
    if child_layer in GENOMIC_LAYERS:
        return False
    if parent_layer == "MIRNA" and child_layer not in MIRNA_TARGETS:
        return False
    return RANK[parent_layer] <= RANK[child_layer]


def _mechanism_edges(rng, layers: np.ndarray, config: MechanismPriorConfig) -> np.ndarray:
    """Adjacency adj[parent, child] of one weakly connected mechanism. Nodes are in layer order (random within a
    layer), so edges only go forward and the graph is acyclic; same-layer edges are allowed."""
    n = len(layers)
    adj = np.zeros((n, n), dtype=bool)
    for child in range(1, n):
        candidates = [p for p in range(child) if _can_parent(layers[p], layers[child])]
        if not candidates or rng.random() < config.root_probability:
            continue
        k = min(int(rng.integers(1, config.max_parents + 1)), len(candidates))
        adj[rng.choice(candidates, k, replace=False), child] = True
    # Connect every node that is still isolated from the others (a root without children, or a node no edge reaches)
    for node in range(n):
        if adj[node].any() or adj[:, node].any() or n == 1:
            continue
        later = [c for c in range(node + 1, n) if _can_parent(layers[node], layers[c])]
        earlier = [p for p in range(node) if _can_parent(layers[p], layers[node])]
        if later:
            adj[node, rng.choice(later)] = True
        elif earlier:
            adj[rng.choice(earlier), node] = True
    return adj


def _sinks(adj: np.ndarray) -> np.ndarray:
    return np.flatnonzero(~adj.any(axis=1))


def _distances_to_sinks(adj: np.ndarray, sinks: np.ndarray) -> np.ndarray:
    """Shortest number of edges from each node to the label, the sinks being its parents (distance 1)."""
    n = adj.shape[0]
    dist = np.full(n, -1)
    dist[sinks] = 1
    frontier = list(sinks)
    while frontier:
        nxt = []
        for node in frontier:
            for p in np.flatnonzero(adj[:, node]):
                if dist[p] == -1:
                    dist[p] = dist[node] + 1
                    nxt.append(p)
        frontier = nxt
    return dist


# ---------------------------------------------------------------------------------------------------------
# Relations: additive functions of the parents


def _function(rng, x: np.ndarray) -> tuple[np.ndarray, str]:
    """One random function of one standardized parent, standardized."""
    kind = rng.choice(FUNCTIONS, p=FUNCTION_PROBS)
    if kind == "linear":
        out = x
    elif kind == "saturating":  # Hill-like activation or repression
        out = np.tanh(rng.uniform(0.5, 3.0) * x + rng.normal(0, 0.5))
    elif kind == "threshold":  # on/off switch at a random quantile
        out = 1 / (1 + np.exp(-rng.uniform(5, 20) * (x - np.quantile(x, rng.uniform(0.2, 0.8)))))
    else:  # non-monotone: a bump around a random centre
        out = np.exp(-((x - rng.normal(0, 0.7)) ** 2) / (2 * rng.uniform(0.3, 1.0) ** 2))
    return _standardize(out[:, None])[:, 0] * rng.choice([-1, 1]), str(kind)


def _sample_roots(rng, n, kind):
    if kind == "normal":
        return rng.normal(0, 1, n)
    if kind == "uniform":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), n)
    if kind == "bimodal":
        return np.where(rng.random(n) < rng.uniform(0.2, 0.8), rng.normal(-1, 0.5, n), rng.normal(1, 0.5, n))
    return rng.standard_t(3, n)  # heavy-tailed


def _node_values(rng, adj, n_rows, config, root_kind):
    """True (latent) values of the nodes of one mechanism in topological (= index) order, standardized, and the
    function of every edge."""
    n = adj.shape[0]
    z = np.zeros((n_rows, n))
    functions = {}
    for node in range(n):
        parents = np.flatnonzero(adj[:, node])
        if not len(parents):
            z[:, node] = _standardize(_sample_roots(rng, n_rows, root_kind)[:, None])[:, 0]
            continue
        signal = np.zeros(n_rows)
        for p in parents:
            out, kind = _function(rng, z[:, p])
            signal += rng.uniform(0.5, 2.0) * out
            functions[(p, node)] = kind
        snr = _loguniform(rng, *config.node_snr)
        noisy = _standardize(signal[:, None])[:, 0] + rng.normal(0, 1 / np.sqrt(snr), n_rows)
        z[:, node] = _standardize(noisy[:, None])[:, 0]
    return z, functions


def _label(rng, parents_z, num_classes, beta, is_null):
    """Class of every row: argmax over classes of additive logits of the label's parents plus Gumbel noise, with
    inverse temperature beta; class biases make the classes unbalanced. A null label ignores the parents."""
    n_rows = parents_z.shape[0]
    functions = {}
    if is_null:
        p = rng.dirichlet(np.full(num_classes, 2.0))
        return rng.choice(num_classes, n_rows, p=p), functions
    logits = np.zeros((n_rows, num_classes))
    for c in range(num_classes):
        for j in range(parents_z.shape[1]):
            out, kind = _function(rng, parents_z[:, j])
            logits[:, c] += rng.uniform(0.5, 2.0) * out
            functions[(j, c)] = kind
    logits = _standardize(logits) * beta + rng.normal(0, 0.5, num_classes)  # beta sets the signal-to-noise
    return np.argmax(logits + rng.gumbel(size=logits.shape), axis=1), functions


# ---------------------------------------------------------------------------------------------------------
# Measurement: true abundance -> measured, processed value of each layer


def _measure(rng, z: np.ndarray, layer: str, variant: str, is_mechanism: np.ndarray | None = None) -> np.ndarray:
    """Observed values of latent columns z (rows, m) of one layer, as the layer's standard processing gives them.
    NaN marks missing values. is_mechanism marks columns in a mechanism (mutations there are drivers)."""
    n, m = z.shape
    if m == 0:
        return np.zeros((n, 0))
    if layer == "SNP":  # genotypes 0/1/2 in Hardy-Weinberg proportions
        maf = rng.uniform(0.05, 0.5, m)
        q0, q1 = (1 - maf) ** 2, 2 * maf * (1 - maf)
        u = _ranks(z)
        return (u > q0).astype(float) + (u > q0 + q1).astype(float)
    if layer == "MUT":  # somatic mutations: rare per gene, a few drivers; mechanism mutations are drivers
        rate = np.clip(10 ** rng.normal(np.log10(0.015), 0.4, m), 0.002, 0.5)
        drivers = rng.random(m) < 0.1
        if is_mechanism is not None:
            drivers |= is_mechanism
        rate = np.where(drivers, rng.uniform(0.1, 0.5, m), rate)
        return (_ranks(z) > 1 - rate).astype(float)
    if layer == "CNV":
        p_neutral = rng.uniform(0.5, 0.9, m)
        p_del = (1 - p_neutral) * rng.uniform(0.2, 0.8, m)
        u = _ranks(z)
        deleted, amplified = u < p_del, u > p_del + p_neutral
        if variant == "discrete":  # GISTIC-like levels -2..2
            deep = rng.uniform(0.1, 0.4, m)
            out = np.zeros((n, m))
            out[deleted] = -1
            out[amplified] = 1
            out[deleted & (u < p_del * deep)] = -2
            out[amplified & (u > 1 - (1 - p_del - p_neutral) * deep)] = 2
            return out
        out = rng.normal(0, 0.05, (n, m))  # continuous log2 ratio: a peak at 0 and tails
        out[deleted] = -(0.4 + 0.5 * np.abs(z[deleted]))
        out[amplified] = 0.4 + 0.6 * np.abs(z[amplified])
        return out
    if layer == "METH":  # beta values; CpG baselines bimodal: mostly near 0 or 1
        kind = rng.choice(3, m, p=[0.45, 0.40, 0.15])
        base = np.where(kind == 0, rng.beta(1, 10, m), np.where(kind == 1, rng.beta(10, 1, m), rng.beta(2, 2, m)))
        base = np.clip(base, 0.01, 0.99)
        logit = np.log(base / (1 - base)) + rng.uniform(0.3, 1.5, m) * z
        return np.clip(1 / (1 + np.exp(-logit)) + rng.normal(0, 0.02, (n, m)), 0, 1)
    if layer in ("RNA", "MIRNA") and variant == "seq":  # NB counts with library size -> log2(CPM + 1)
        if layer == "RNA":
            base = rng.normal(np.log(30), 2.0, m)  # log CPM of each gene: several orders of magnitude
            library = _loguniform(rng, 2e6, 4e7) * np.exp(rng.normal(0, 0.3, n))
        else:  # few miRNAs take most counts, many are barely detected
            base = rng.normal(np.log(5), 3.0, m)
            library = _loguniform(rng, 5e5, 1e7) * np.exp(rng.normal(0, 0.4, n))
        log_cpm = base + rng.uniform(0.3, 1.5, m) * z
        mean = library[:, None] / 1e6 * np.exp(np.clip(log_cpm, -10, 12))
        dispersion = rng.uniform(0.02, 0.2) + 1 / np.sqrt(np.exp(base) + 1)  # larger for low expression
        shape = 1 / dispersion
        counts = rng.poisson(rng.gamma(shape, mean / shape))
        return np.log2(counts / library[:, None] * 1e6 + 1)
    if layer in ("RNA", "MIRNA"):  # microarray: log2 intensity with background and saturation
        base = rng.normal(8, 2, m)
        return np.clip(base + rng.uniform(0.3, 1.5, m) * z + rng.normal(0, 0.2, (n, m)), 2, 16)
    # PROT, METAB: log intensities; mass spectrometry loses low values (below a detection limit)
    base = rng.normal(20 if layer == "PROT" else 15, 2, m)
    x = base + rng.uniform(0.3, 1.5, m) * z + rng.normal(0, 0.2, (n, m))
    if variant == "rppa":
        return x
    rate = rng.uniform(0.1, 0.5) if layer == "PROT" else rng.uniform(0.05, 0.3)
    centered = (x - base) / x.std(axis=0).clip(1e-6)  # missingness depends on abundance (column scale)
    abundance = centered + (base - base.mean()) / max(base.std(), 1e-6)  # low-abundance columns lose more
    limit = np.quantile(abundance, rate)
    missing = rng.random((n, m)) < 1 / (1 + np.exp((abundance - limit) / 0.3))
    missing |= rng.random((n, m)) < (0.02 if layer == "METAB" else 0.0)
    return np.where(missing, np.nan, x)


def _variants(rng, layers) -> dict:
    """Measurement variant of each active layer, drawn once per table."""
    out = {}
    for layer in map(str, layers):
        if layer == "RNA":
            out[layer] = "seq" if rng.random() < 0.7 else "array"
        elif layer == "MIRNA":
            out[layer] = "seq" if rng.random() < 0.8 else "array"
        elif layer == "CNV":
            out[layer] = "discrete" if rng.random() < 0.6 else "continuous"
        elif layer == "PROT":
            out[layer] = "ms" if rng.random() < 0.6 else "rppa"
        elif layer == "METAB":
            out[layer] = "ms"
        else:
            out[layer] = "default"
    return out


DISCRETE = {"SNP": True, "MUT": True}


def _is_discrete(layer, variant) -> bool:
    return DISCRETE.get(layer, False) or (layer == "CNV" and variant == "discrete")


# ---------------------------------------------------------------------------------------------------------
# Widening: copies with less information, noise, constants


def _copy_latent(rng, z_sources: np.ndarray, kind: str, config) -> np.ndarray:
    """Latent value of one copy: a (non-)linear function of its sources mixed with its own noise, correlation
    rho < 1 with the function, so that the copy carries less information than its sources."""
    n = z_sources.shape[0]
    if kind == "linear":
        signal = z_sources @ (rng.uniform(0.5, 1.5, z_sources.shape[1]) * rng.choice([-1, 1], z_sources.shape[1]))
    else:
        x = z_sources[:, 0]
        f = rng.choice(["exp", "sigmoid", "abs", "square"])
        signal = {"exp": np.exp(rng.uniform(0.3, 1.0) * x), "sigmoid": 1 / (1 + np.exp(-rng.uniform(1, 4) * x)),
                  "abs": np.abs(x), "square": x**2}[f]  # fmt: skip
    signal = _standardize(signal[:, None])[:, 0]
    rho = rng.uniform(*config.copy_correlation)
    return rho * signal + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)


def _constant_column(rng, n, layer, variant):
    """A constant column, or a near-constant one (all but a few rows equal), on the layer's scale."""
    value = {"SNP": 0.0, "MUT": 0.0, "CNV": 0.0, "METH": rng.uniform(0, 1)}.get(layer, rng.uniform(0, 10))
    col = np.full(n, value)
    if rng.random() < 0.5:  # near-constant
        k = max(1, int(rng.uniform(0.01, 0.05) * n))
        rows = rng.choice(n, k, replace=False)
        if layer in ("SNP", "MUT", "CNV"):
            col[rows] = value + 1.0  # one level up: still a valid genotype, mutation or copy-number level
        elif layer == "METH":
            col[rows] = np.clip(value + rng.normal(0, 0.2, k), 0, 1)  # beta values stay in [0, 1]
        else:
            col[rows] = value + rng.normal(0, 1, k)
    return col


# ---------------------------------------------------------------------------------------------------------
# One table


def sample_dataset(rng: np.random.Generator, config: MechanismPriorConfig | None = None,
                   n_rows: int | None = None) -> MechanismDataset:  # fmt: skip
    """Samples one table with its ground truth. n_rows fixes the number of rows (to batch tables together)."""
    config = config or MechanismPriorConfig()
    for _ in range(100):
        rows = n_rows or _loguniform_int(rng, config.min_rows, config.max_rows)
        table = _try_sample(rng, config, rows)
        if table is not None:
            return table
    raise RuntimeError("could not sample a table with every class present")


def _try_sample(rng, config, n_rows):
    active = _sample_layers(rng)
    variant = _variants(rng, active)
    num_classes = _sample_num_classes(rng, n_rows, config)
    root_kind = rng.choice(["normal", "uniform", "bimodal", "heavy"])

    # Mechanisms: the label's (index 0), the others, the independent variables (mechanisms of one variable)
    sizes = [_loguniform_int(rng, *config.label_mechanism_size)]
    low, high = config.num_other_mechanisms
    sizes += [_loguniform_int(rng, *config.other_mechanism_size) for _ in range(int(rng.integers(low, high + 1)))]
    n_independent = int(rng.integers(config.num_independent[0], config.num_independent[1] + 1))
    sizes += [1] * n_independent

    columns_z, columns_layer, roles, distances, mech_ids, is_mech = [], [], [], [], [], []
    edges = []  # (parent node, child node or "label", function), node = graph column index
    label_parents_z, label_parent_nodes = None, None
    offset = 0
    for m, size in enumerate(sizes):
        layers = _mechanism_layers(rng, size, active)
        adj = _mechanism_edges(rng, layers, config) if size > 1 else np.zeros((1, 1), dtype=bool)
        z, functions = _node_values(rng, adj, n_rows, config, root_kind)
        if m == 0:
            sinks = _sinks(adj)  # the label's parents: every node then has a path to the label
            dist = _distances_to_sinks(adj, sinks)
            label_parents_z, label_parent_nodes = z[:, sinks], sinks + offset
            role = np.full(size, LABEL_MECHANISM)
        else:
            dist = np.full(size, -1)
            connected = adj.any(axis=0) | adj.any(axis=1)  # e.g. genomic-only: no edges are possible
            role = np.where(connected, OTHER_MECHANISM, INDEPENDENT)
        edges += [(offset + p, offset + c, f) for (p, c), f in functions.items()]
        columns_z.append(z)
        columns_layer += list(layers)
        roles += list(role)
        distances += list(dist)
        mech_ids += list(np.where(role == INDEPENDENT, -1, m))
        is_mech += [True] * size  # graph variables: their mutations are drivers, never all-zero
        offset += size

    is_null = rng.random() < config.null_probability
    beta = _loguniform(rng, *config.label_beta)
    y, label_functions = _label(rng, label_parents_z, num_classes, beta, is_null)
    if np.bincount(y, minlength=num_classes).min() < 2:
        return None
    for j, node in enumerate(label_parent_nodes):  # one function per parent and class
        kinds = sorted({f for (jj, _), f in label_functions.items() if jj == j})
        edges.append((int(node), "label", "+".join(kinds) if kinds else "none"))

    z_graph = np.column_stack(columns_z)
    layer_graph = np.array(columns_layer)
    n_graph = z_graph.shape[1]

    # Widening
    # total width log-uniform up to the maximum (no widening if the graph alone reaches it)
    n_total = (
        n_graph if n_graph >= config.max_total_features else _loguniform_int(rng, n_graph, config.max_total_features)
    )
    n_new = n_total - n_graph
    # constants: an absolute share; copies and noise share the rest in proportion to their draws
    constant = rng.uniform(*config.constant_fraction)
    rest = np.array([rng.uniform(*config.copy_linear_fraction), rng.uniform(*config.copy_nonlinear_fraction),
                     rng.uniform(*config.noise_fraction)])  # fmt: skip
    rest = rest / rest.sum() if rest.sum() > 0 else np.array([0.0, 0.0, 1.0])
    counts = rng.multinomial(n_new, np.append(rest * (1 - constant), constant))
    new_z, new_layer, new_role, new_sources, new_kind, constants = [], [], [], [], [], []
    for kind, count in zip(("linear", "non_linear"), counts[:2], strict=True):
        for _ in range(count):
            first = int(rng.integers(n_graph))
            same_layer = np.flatnonzero(layer_graph == layer_graph[first])  # the sources share a layer
            k = int(rng.integers(1, 4)) if kind == "linear" else 1
            others = rng.permutation(same_layer[same_layer != first])[: k - 1]
            sources = np.concatenate([[first], others]).astype(int)
            new_z.append(_copy_latent(rng, z_graph[:, sources], kind, config))
            # measured in the layer of its (first) source: a copy measured by a richer technology (a continuous
            # layer copying a binary mutation) would carry more information than its source as observed
            new_layer.append(layer_graph[sources[0]])
            new_role.append(COPY)
            new_sources.append(sorted(int(s) for s in sources))
            new_kind.append(kind)
    for _ in range(counts[2]):
        new_z.append(rng.normal(0, 1, n_rows))
        new_layer.append(rng.choice(active))
        new_role.append(NOISE)
        new_sources.append([])
        new_kind.append("")
    for _ in range(counts[3]):
        layer = rng.choice(active)
        constants.append((layer, _constant_column(rng, n_rows, layer, variant[layer])))

    # Measurement, layer by layer
    all_z = np.column_stack([z_graph] + ([np.column_stack(new_z)] if new_z else []))
    all_layer = np.concatenate([layer_graph, np.array(new_layer, dtype=layer_graph.dtype)])
    mech_flags = np.concatenate([np.array(is_mech), np.zeros(len(new_z), bool)])
    X = np.zeros((n_rows, all_z.shape[1]))
    for layer in active:
        idx = np.flatnonzero(all_layer == layer)
        X[:, idx] = _measure(rng, all_z[:, idx], layer, variant[layer], mech_flags[idx])
    if constants:
        X = np.column_stack([X] + [c for _, c in constants])
        all_layer = np.concatenate([all_layer, np.array([la for la, _ in constants], dtype=all_layer.dtype)])

    n_cols = X.shape[1]
    role = np.array(roles + new_role + [CONSTANT] * len(constants))
    distance = np.array(distances + [-1] * (n_cols - n_graph))
    mechanism = np.array(mech_ids + [-1] * (n_cols - n_graph))
    sources = [[] for _ in range(n_graph)] + new_sources + [[] for _ in constants]
    kinds = np.array([""] * n_graph + new_kind + [""] * len(constants), dtype=object)

    # Shuffle the columns; remap the sources and the edges to the new column indices
    perm = rng.permutation(n_cols)
    new_index = np.empty(n_cols, dtype=int)
    new_index[perm] = np.arange(n_cols)
    return MechanismDataset(
        X=X[:, perm],
        y=y,
        layer=all_layer[perm],
        is_discrete=np.array([_is_discrete(la, variant.get(la)) for la in all_layer[perm]]),
        role=role[perm],
        distance=distance[perm],
        mechanism=mechanism[perm],
        copy_sources=[[int(new_index[s]) for s in sources[i]] for i in perm],
        copy_kind=kinds[perm],
        num_classes=num_classes,
        is_null=bool(is_null),
        edges=[(int(new_index[p]), c if c == "label" else int(new_index[c]), f) for p, c, f in edges],
        variant=variant,
    )
