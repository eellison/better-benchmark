# Matmul pattern corpus (`mm_repro/`)

`mm_repro/` holds small benchmark repros cut out of real model graphs, one
fusion pattern around one to three matmuls (MMs) each, sorted by pattern
category so results can be read per category.

## Regenerate

From the repository root, with a compatible CUDA-enabled PyTorch nightly:

```bash
python scripts/extract_matmul_corpus.py --workers 32   # writes mm_repro/
python scripts/validate_matmul_repros.py               # FakeTensor-runs every shape point
pytest tests/test_matmul_partitioner.py tests/test_mm_*.py
```

- Input: the saved `repros/models/**/full_graph_*.py` graphs and their
  `.meta.json` sidecars. Nothing else is read.
- Everything runs on CPU with FakeTensors; nothing is compiled. With 32
  workers the whole run takes about two minutes.
- Only `mm_repro/` is written, and it is replaced atomically, so files from an
  older run never survive. Intermediate files live in a temporary directory.
- The output is deterministic: the same sources and code give an identical
  `mm_repro/`, independent of worker count.
- Options: `--output DIR`, `--workers N`, `--source FILE` (repeatable).

Current snapshot: **585 repros / 1019 shape points** from 13275 region
occurrences in 364 source graphs, one repro per pattern family (377 of the
962 curated repros were folded into their family representative).

| category | repros | before dedup | shape points |
| --- | --- | --- | --- |
| prologue | 49 | 107 | 91 |
| epilogue | 284 | 518 | 602 |
| prologue_epilogue | 79 | 112 | 104 |
| horizontal | 74 | 118 | 112 |
| horizontal_combine | 5 | 5 | 7 |
| vertical | 94 | 102 | 103 |

## Categories

[`matmul_partitioner.py`](../matmul_partitioner.py) extracts every category
separately (`get_matmul_regions`):

| category | contents |
| --- | --- |
| `prologue` | one MM; pointwise/gather/transparent work producing its operands, nothing after it |
| `epilogue` | one MM; pointwise/scatter/transparent work on its result (side inputs, branches, several outputs allowed), nothing in front |
| `prologue_epilogue` | one MM with both |
| `horizontal` | 2-3 independent MMs sharing an operand through transparent ops, equal output shapes (K may differ), optional prologues/epilogues; every output depends on one MM only |
| `horizontal_combine` | the same, but a pointwise op combines >= 2 of the MMs, e.g. SwiGLU `silu(x@wg) * (x@wu)` |
| `vertical` | >= 2 dependent reduction stages (MMs and ordinary reductions), >= 1 MM, accepted by the slice analysis below |

- The single-MM categories are exclusive: an MM with work on both sides is
  only `prologue_epilogue`. Each horizontal group is likewise either
  `horizontal` or `horizontal_combine`.
- Transparent ops (view, reshape, permute, transpose, expand, squeeze,
  casts, ...) are always included and never count as work.
- Outside `vertical`, ordinary reductions and other MMs are boundaries: their
  results become inputs.
- Horizontal groups: MMs are grouped by (shared input, output shape); a group
  takes the first independent MMs in graph order and stops at 3. Joins of MMs
  that share no operand (e.g. GPT-J's parallel residual) are not groups.
- Side inputs (an epilogue's residual, an addmm bias) are pulled in through
  transparent and gather ops, and when computed only from graph inputs and
  constants (dropout masks, iota masks, gelu' of a saved activation).
- Regions must replay standalone: no value leaves a region and flows back
  into it, and no outside op writes memory it uses while it runs. Unbind,
  softmax, attention ops and copies to another device are boundaries.
- Single-MM regions whose only extra work is padding/cropping, packing the
  addmm bias, copying a whole input or a lone `bmm == -inf` are dropped
  ([`mm_region_quality.py`](../mm_region_quality.py)).
- Regions may overlap and the model graph is never modified. Selection uses
  op schemas, value dependencies and tensor metadata only.

Size limits of a region (`MatmulLimits`): 3 MMs, 48 FX nodes, 32 MMs plus
real computation, 12 inputs, 4 outputs, 8 reduction stages on one path.

## Vertical slice analysis

Every MM is a reduction (`H[m,n] = sum_k A[m,k] B[k,n]`, reduced axis K).
Tensor axes are tracked as union-find atoms with sizes: broadcasting,
views/reshapes (axis split and merge), permutes and MM batch/row/column axes
unify atoms; partial slices, gathers and unknown ops create fresh ones, and
values computed only from region inputs get fresh atoms at every use. An atom
is an independent slice axis if it appears in every region output and in
every internal stage result that output depends on.

For every stage:

- `R` = product of the reduced axis lengths, `R <= 128`, including the first
  MM and reductions over external inputs.
- `F` = elements of the stage result needed per slice by later stages (0 if
  none), `F <= 128`.

128 is one Triton tile without padding. Hand-written single-kernel Helion
versions of the vertical repros showed that regions at R/F = 144-192 (padded
to 256) were not faster as one kernel
([vertical_fusion_study.md](vertical_fusion_study.md)).

Stages are absorbed greedily in graph order and the analysis is re-run for
every trial. Candidates are rejected when an internal index relation is
unknown: a gather, scatter, cat or unknown op reading an internal stage
result, a reshape that does not align, or an axis appearing twice in one
tensor. External table gathers are fine. Each vertical repro's `shapes.json`
records `partition_metrics.vertical`: the slice axis sizes and every stage's
R and F.

| region | result |
| --- | --- |
| `(A@B).view(M,128,32).sum(-1)`, K=128 | accept: slice (m, q), R = 128/32, F = 32 |
| `A@B@C@D`, widths 128/64/128/64 | accept: slice m, F = 64/128 |
| `relu(A@B) @ v`, N=256 | reject: F = 256 |
| gather -> LN(128) -> MM -> LN(256) | only LN -> MM (the second LN has R = 256) |
| `A@B@C`, K=512 | reject: first MM R = 512 |

## How a region becomes a repro

1. **Capture** (`scripts/extract_matmul_corpus.py`): the regular capture hook
   code extracts each region with its explicit outputs into a temporary raw
   corpus. An input that reaches a `sqrt`/`rsqrt` only through views, casts
   or adding a non-negative epsilon gets a positive generator
   (`abs(randn) + 0.1`), so replays do not produce NaNs.
2. **Curate** (`scripts/curate_mm_corpus.py`): raw repros over the size limits
   are skipped. Each shape point is re-traced and normalized only where
   tensor metadata proves the result unchanged (input permutations become
   strided inputs, identity views and contiguous reshape chains collapse,
   reduction axes/keepdim are canonical, identical internal pointwise ops
   merge), dropout seed slots are set to slot 0, and the pattern category is
   re-derived from the graph. Points whose DAGs are isomorphic, including
   every scalar literal and commutative operand order, share one repro if
   replaying it on the point's inputs reproduces the same output metadata.
3. **Families**: repros are grouped by `structure_key`, a hash of the compute
   DAG plus the input/output dtypes that ignores transparent ops and scalar
   literals. Only the member with the most source occurrences is exported.
   This drops layout and scalar variants and the shape points of the other
   members; `families.json` lists them.
4. **Publish**: the curated tree replaces `mm_repro/`.

Real layouts, storage offsets, aliasing between inputs, casts and scalar
values of the exported repros are preserved. Pattern hashes are
scalar-sensitive (`pattern_hash_mode: scalar_literals_v1`).

## What is in `mm_repro/`

| Path | Contents |
| --- | --- |
| `canonical/<category>/<mm ops>_<hash>/repro.py` | FX code of the region (`class Repro`) plus `make_inputs()` |
| `canonical/<category>/<mm ops>_<hash>/shapes.json` | every shape point: input shapes, dtypes, strides, offsets, generators, source models, partition metrics |
| `canonical/<category>/<mm ops>_<hash>/meta.json` | pattern hash, category, family, applied normalizations, coverage labels |
| `families.json` | every family: representative, members (hash, occurrences, shape points) |
| `models/<suite>/<mode>/<model>/manifest.json` | the repros found in each model |
| `occurrences.json` | per source graph: each region (source node names, outputs) and the repro/shape it became; folded family members carry `family_member` and `shape_hash: null` |
| `curation_mapping.json` | raw capture (pattern, shape) -> exported (pattern, shape), with input order and views |
| `inventory.json`, `report.json`, `validation.json` | summary, generation settings, FakeTensor results |

## Capturing MM regions from other models

```python
from capture_hook import install_capture_hook

install_capture_hook("out_dir", only_include_mm=True)
```

or set `REPRO_CAPTURE_ONLY_INCLUDE_MM=1` next to `REPRO_CAPTURE_DIR`.

## Limits of this corpus

Selection does not certify that fusing a region is faster; that needs real
compilation and measurement. For the vertical category,
[vertical_fusion_study.md](vertical_fusion_study.md) measured hand-written
Helion kernels: every written vertical repro became one kernel without
recomputation, but regions with few independent slices (e.g.
`bmm_c044adeba521`, 16 slices) are still slower as one kernel.
