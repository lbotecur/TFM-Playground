# TabOmics: design of the public repository

Agreed on 2026-10-13, before writing any of it. TFM-Playground stays the development repository (experiments,
branches, diagnostics); TabOmics is the clean package people install and use.

## Goals

1. **Use the model:** `pip install tabomics`, the v2 checkpoint downloaded from Hugging Face, a classifier and a
   regressor with a scikit-learn interface, embeddings and feature importance.
2. **Evaluate any model** on our benchmarks, with fixed folds, results saved to CSV.
3. **Train:** from scratch or continuing a checkpoint, with our prior and the exact v2 recipe.
4. Clean, commented code; only what is used. Apache 2.0, crediting nanoTabPFN (Pfefferle et al.) for the
   original architecture and TabICL (Inria Soda team) for the graph generator.

## Names

- Package and model: **TabOmics** (`tabomics`, free on PyPI on 2026-10-13).
- Architecture: `TabOmicsModel`, derived from nanoTabPFN (credited in README, NOTICE and the paper, not in the
  code names). The weights of v2 load unchanged: a checkpoint stores the names of the inner layers
  (`feature_encoder.linear_layer.weight`, ...), not the class name; a parity test checks it.
- Public API: `tabomics.Classifier`, `tabomics.Regressor`.

## Layout

```
tabomics/
├── src/tabomics/
│   ├── model/          # TabOmicsModel: attention between columns and rows, missing-value channel,
│   │                   # feature clip, low-memory inference, column vectors
│   ├── inference/      # Classifier, Regressor, preprocessing (ordinal codes, constant columns),
│   │                   # column permutation, embeddings, feature importance (label attention)
│   ├── hub.py          # model names -> Hugging Face repo + commit + sha256; download with cache
│   ├── prior/
│   │   ├── graph/      # TabICL's graph generator (graph_lib + batching), copied with its licence and a
│   │   │               # CHANGES.md listing every change we make to it
│   │   ├── raw.py      # graph_scm_raw: no clamp, no z-score, categorical masks
│   │   ├── widening.py # TabPFN-Wide's algorithms 1-2 (our implementation), rows by a rows x columns budget
│   │   ├── sparse.py   # sparse columns
│   │   └── loader.py   # batches, workers, reproducible seeds (PYTHONHASHSEED)
│   ├── training/       # training loop, configuration, export of releases
│   └── benchmarks/     # hdlss, mlomics, shamir, cc18: loading, fixed folds, metrics, runner
├── scripts/
│   ├── train.py        # from scratch, --init-from, --resume
│   ├── export.py       # training checkpoint -> release (safetensors + config.json)
│   └── evaluate.py     # --model v2|path --suite hdlss mlomics shamir cc18
├── configs/training/   # the v2 recipe, stage by stage (YAML)
├── benchmarks/download/# scripts to download the benchmark data (not redistributed)
├── checkpoints/v2/     # model card: recipe, config, sha256, Hugging Face link (no weights in git)
├── examples/           # quickstart, wide omics data, embeddings, feature importance, own checkpoint,
│                       # continue training, regression
├── tests/              # unit tests, parity with TFM-Playground, reproduction of v2
├── docs/               # method, prior, training, evaluation
└── LICENSE · NOTICE · CITATION.cff · README.md · pyproject.toml
```

One install: `pip install tabomics` brings everything (inference, prior, training, benchmarks). No TabICL
dependency: the part of its generator we use is in `prior/graph/`.

## Using a model

```python
from tabomics import Classifier
clf = Classifier()                         # v2 from Hugging Face
clf = Classifier(model="v2")               # the same, explicit
clf = Classifier(model="my_model.ckpt")    # a local file, or a path from hf_hub_download(...)
clf.fit(X_train, y_train).predict_proba(X_test)
```

- Weights in **safetensors** (only tensors: unlike a pickled `.pth`, loading cannot run code) plus a
  `config.json` with the architecture and how the model is used (feature clip 10, low-memory inference, column
  permutation off for v2). Nobody has to know these settings to get the paper's numbers.
- Download into Hugging Face's cache (`~/.cache/huggingface/hub`; `HF_HOME` or `cache_dir=` to move it), once;
  offline afterwards.
- **The default model is fixed by the package version** (v2 in 1.x): results never change silently. A new model
  comes with a new package version (`pip install -U tabomics`) or by asking for it by name.
- Each model name points to a fixed Hugging Face commit and sha256, never to "the latest file".
- Hugging Face: account `lbotecur`, one model repository per release (`lbotecur/tabomics-v2`), private until
  the paper is out.
- Regressor: same interface; its default model is to be decided (we have no regressor of our own yet).

## Benchmarks

```bash
python scripts/evaluate.py --model v2 --suite hdlss mlomics
python scripts/evaluate.py --model my_model.ckpt --suite cc18
```

Evaluates only the model given, on the fixed folds of each suite, and saves one row per fold to CSV plus a
summary per suite. It does not compare with other models: each user tunes baselines as they see fit. The
paper says which suites it reports.

## Reproducibility

- A test checks that v2 gives the AUROC we report on a sample of folds (tolerance 1e-6: bf16 on GPU is not
  bit-exact across hardware; versions of torch, CUDA and the GPU are recorded).
- A test checks that our copy of the generator, unchanged, gives the same tables as TabICL 2.2.0, so the
  priors of v2 and of the v3 runs can be regenerated.
- Training recipes fix the seed and PYTHONHASHSEED.

## Keeping it in step with TFM-Playground

By hand: what is validated in TFM-Playground is ported to TabOmics with a pull request and its tests, cleaned
on the way. Parity tests (same checkpoint and data, same predictions in both repositories) catch a change
that was not ported. What is not used stays out (omics prior, mlp/tree/mix priors, ticl/tabpfn priors, the
inherited nanoTabPFN scripts, diagnostics, comparison scripts); what is used is chosen by running the v2
reproduction with coverage, not by eye.

## Order of work

1. `model/`, `inference/`, `hub.py`, the conversion of v2 to safetensors; parity tests.
2. `benchmarks/` and `evaluate.py`.
3. `prior/` (with the copied generator and its identity test), `training/`, the v2 recipe.
4. Examples, docs, model card.
5. v2 on Hugging Face, package published.

## Open

- Default model of the regressor: the original nanoTabPFN regressor (saying it is not ours) or a required path.
- GitHub repository private until the paper is submitted (recommended).
