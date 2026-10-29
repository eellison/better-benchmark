# Matmul fusion corpus (`mm_repro/`)

`mm_repro/` holds small benchmark repros cut out of real model graphs. Each one
is an MM (or a few MMs) plus the work around it that a compiler could fuse into
the same kernel, e.g. a SwiGLU gate, attention-score projections, or a
normalization in front of a classifier.

## Regenerate

From the repository root, with a compatible CUDA-enabled PyTorch nightly:

```bash
python scripts/extract_matmul_corpus.py      # writes mm_repro/
python scripts/validate_matmul_repros.py     # FakeTensor-runs every shape point
```

- Input: the saved `repros/models/**/full_graph_*.py` graphs and their
  `.meta.json` sidecars. Nothing else (no handwritten patterns, no older
  corpus) is read.
- Everything runs on CPU with FakeTensors; nothing is compiled. With
  `--workers 16` on a large machine the whole run takes about a minute.
- Only `mm_repro/` is written, and it is replaced atomically, so files from an
  older run never survive. Intermediate files live in a temporary directory.
- The output is deterministic: the same sources and code give a byte-identical
  `mm_repro/`, independent of worker count.
- Current snapshot: **276 repros / 501 shape points**, from 4671 region
  occurrences in 364 source graphs. These numbers are a result, not a quota.

Options: `--output DIR`, `--workers N`, and `--source FILE` (repeatable) to
process only some graphs.

## What is in `mm_repro/`

| Path | Contents |
| --- | --- |
| `canonical/<mm ops>_<hash>/repro.py` | FX code of the region (`class Repro`) plus `make_inputs()` |
| `canonical/<mm ops>_<hash>/shapes.json` | every shape point: input shapes, dtypes, strides, offsets, generators, source models |
| `canonical/<mm ops>_<hash>/meta.json` | pattern hash, applied normalizations, coverage labels |
| `models/<suite>/<mode>/<model>/manifest.json` | the repros found in each model |
| `occurrences.json` | per source graph: each region (source node names, outputs) and the repro/shape it became |
| `curation_mapping.json` | raw capture (pattern, shape) → exported (pattern, shape), with input order and views |
| `inventory.json`, `report.json`, `validation.json` | summary, generation settings, FakeTensor results |

Examples:

- [SwiGLU gate](../mm_repro/canonical/mm_30a54d87fa25/repro.py): gate and up
  projections, SiLU, multiply (Qwen3, Mistral).
- [Attention scores](../mm_repro/canonical/addmm_bmm_e2c829b4a8ca/repro.py):
  Q/K projections, scale and transpose, then a BMM (BERT family).
- [Norm → classifier → log-softmax](../mm_repro/canonical/addmm_ebba12fa4dcc/repro.py):
  a reduction before and after the same MM (maml).
- [Two independent MMs](../mm_repro/canonical/mm_39e6a3393286/repro.py): a
  horizontal pair sharing an operand.

## How regions are selected

`get_fusion_partitions(gm, only_include_mm=True)` calls
`matmul_partitioner.get_matmul_regions`. The comment at the top of
[`matmul_partitioner.py`](../matmul_partitioner.py) explains the rules and the
search in detail. In short:

1. At most 5 reductions (MMs and ordinary reductions together) on any one
   dependency path, enough for "norm → MM → norm".
2. An ordinary reduction may reduce at most 128 elements per output; larger
   ones become region inputs. The first MM of a chain may have any K, a later
   MM needs K ≤ 128.
3. Independent MMs must produce the same output shape (K may differ).
4. Every output needs an MM and real computation; views, casts and copies do
   not count. A group of independent MMs is useful on its own.
5. A single MM whose only extra work is padding/cropping, packing the addmm
   bias, a whole-input copy, or `bmm == -inf` is dropped
   ([`mm_region_quality.py`](../mm_region_quality.py)).
6. Regions must replay standalone: unbind, softmax, attention, grouped MM and
   device copies are boundaries, and no value may leave and re-enter a region.

Regions may overlap, and the model graph is never modified. The search uses
op schemas, value dependencies and tensor metadata only: no model names, no
pattern templates, no ranking or sampling.

Size limits while searching, and for an exported repro:

| | search (`MatmulLimits()`) | exported repro |
| --- | --- | --- |
| MMs | 4 | 3 |
| FX nodes | 96 | 42 |
| MMs + real computation | 48 | 32 |
| inputs | 24 | 8 |
| outputs | 4 | 2 |
| reductions on one path | 5 | 5 |
| elements per reduction, K of a chained MM | 128 | 128 |

## How a region becomes a repro

1. **Capture** (`scripts/extract_matmul_corpus.py`): the regular capture hook
   code extracts each region with its explicit outputs into a temporary raw
   corpus. An input that reaches a `sqrt`/`rsqrt` only through views, casts
   or adding a non-negative epsilon (e.g. an externally computed variance)
   gets a positive generator (`abs(randn) + 0.1`), so replays do not produce
   NaNs.
2. **Curate** (`scripts/curate_mm_corpus.py`): raw repros over the exported
   limits are skipped. Each shape point is re-traced and normalized, but only
   where tensor metadata proves the result unchanged: input permutations
   become strided inputs (also through float casts), identity views and
   contiguous reshape chains collapse, reduction axes/keepdim are canonical,
   identical internal pointwise ops merge, and identity adaptive pooling is
   removed. Points whose DAGs are isomorphic, including every scalar literal
   and commutative operand order, share one repro if replaying that repro on
   the point's inputs reproduces the same output metadata.
3. **Publish**: the curated tree replaces `mm_repro/`.

Real layouts, storage offsets, aliasing between inputs, casts and scalar
values are always preserved.

MM pattern hashes are scalar-sensitive (`pattern_hash_mode:
scalar_literals_v1`): numeric literals, numeric keyword arguments and tuple
indices are part of the identity, because the repros bake them in. Pattern
hashes of the existing non-MM corpus are unchanged.

## Capturing MM regions from other models

```python
from capture_hook import install_capture_hook

install_capture_hook("out_dir", only_include_mm=True)
```

or set `REPRO_CAPTURE_ONLY_INCLUDE_MM=1` next to `REPRO_CAPTURE_DIR`.

## Tests

```bash
pytest tests/test_matmul_partitioner.py tests/test_mm_*.py
```

`tests/test_mm_corpus_snapshot.py` checks the committed snapshot itself.

## Limits of this corpus

Selection does not certify that one GPU kernel can run a region or that fusing
it is faster; that needs real compilation and measurement. Only one
reduction → MM → reduction region exists (norms that reduce more than 128
elements are boundaries by rule 2), and the exact gather → LayerNorm → MM →
LayerNorm combination does not occur in the saved graphs.
