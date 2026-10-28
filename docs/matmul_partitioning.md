# Small matmul benchmark regions

```python
from capture_hook import get_fusion_partitions, install_capture_hook
from matmul_partitioner import MatmulLimits

regions = get_fusion_partitions(gm, only_include_mm=True)
install_capture_hook("mm_repro", only_include_mm=True)
```

Other model repositories can set `REPRO_CAPTURE_ONLY_INCLUDE_MM=1` alongside
`REPRO_CAPTURE_DIR`.

## Reproduce the corpus

Run from the repository root with a compatible PyTorch installation:

```bash
python scripts/extract_matmul_corpus.py --output mm_repro --workers 4
python scripts/validate_matmul_repros.py mm_repro
```

The input corpus is the saved `repros/models/**/full_graph_*.py` files and their
metadata sidecars. Loading uses `full_graph_harness` and `roundtrip_validation`;
partition extraction and serialization use `capture_hook` and `merge_captures`.
The collector uses FX value dependencies, operator schemas/tags and tensor
metadata. No external model repository or handwritten pattern corpus is required.
Tracing and validation run with FakeTensor, without GPU compilation or autotuning.

Each unique DAG is saved under `mm_repro/canonical/<aten_names>_<hash>/` with
`repro.py`, `meta.json` and `shapes.json`. `report.json`, `inventory.json` and
`occurrences.json` record aggregate statistics and source provenance;
`validation.json` records reload results. A shape point is a distinct input
shape/stride/dtype configuration for one deduplicated repro.

Intermediate captures are removed after successful extraction. Use `--keep-cache`
to retain them for debugging, or `--resume` to continue an interrupted run.
For one saved graph, pass `--source path/to/full_graph_000.py`.

## Current examples

- [Four-MM chain](../mm_repro/canonical/addmm_8ed26adc744a/repro.py):
  16-wide intermediate matrices with tanh between contractions.
- [Horizontal projections into a BMM](../mm_repro/canonical/addmm_bmm_fe4891df3e8d/repro.py):
  two 128-wide projections, then a BMM contracting 32 features.
- [SwiGLU gating](../mm_repro/canonical/mm_bdd1d6cf4c6a/repro.py):
  two MMs plus SiLU/multiplication, ending before a large down projection.
- [MM plus GELU and layer normalization](../mm_repro/canonical/addmm_4fb63f9ccc45/repro.py):
  M=32768, K=256, N=128, with row statistics and normalization.

See [the corpus summary](../mm_repro/README.md) for final statistics.

## Selection

Each MM seeds a bounded local dataflow search. Supported pointwise, indexing,
views and reductions grow forward; joins bring in sibling MM branches.
Upstream prologue cones are tried atomically. A rejected next contraction
leaves the producer MMs and intervening pointwise computation as a candidate.
There are no model names or registered operator sequences.
When a supported value path contains an earlier MM, its intervening PW stays
with that producer's epilogue. It is not collected again as the next MM's
prologue. Input-side gather/scaling/math without such an upstream MM can still
be genuine prologues. This avoids redundant `MM1→PW` and `PW→MM2` benchmarks
when a large handoff prevents collecting the whole chain.
Neighboring independent candidates also merge horizontally when their MM row
domains, outputs and contraction budgets are compatible. Shared inputs and
an explicit combine node are not required.

An output qualifies only if its value cone contains both an internal MM and
real non-MM computation. Casts, views, copies, allocations and scalar identities
do not qualify. Nodes needed only by rejected outputs are removed. The source
model graph stays intact: these independent microbenchmarks may overlap and
do not form a replacement execution schedule for the model.
Pure horizontal regions with at least two independent MMs are also retained:
`x @ w, x @ y` is a useful fusion task without additional pointwise computation.
Bare MM outputs are otherwise excluded.

## Eight limits

Repro directory prefixes use actual ATen operation names: `mm_<hash>`,
`addmm_<hash>`, `bmm_<hash>`, or a combination such as `addmm_bmm_<hash>`.
The DAG hash already distinguishes the concrete MM operations.

| Limit | Default | Purpose |
|---|---:|---|
| `max_mm` | 4 | Total matrix contractions, including horizontal branches |
| `max_nodes` | 96 | Total FX region size |
| `max_compute` | 48 | MM plus substantive non-MM computation |
| `max_outputs` | 4 | Meaningful benchmark outputs |
| `max_inputs` | 24 | Benchmark arguments |
| `max_contraction_depth` | 4 | MM and ordinary reductions on any dependency path |
| `max_internal_dim` | 128 | Handoffs, linked MM feature dimensions, individual reduction axes |
| `max_reduction_extent` | 512 | Cumulative ordinary reduction extent and initial MM K before feature collapse |

Standalone/horizontal MMs may have large external K and output columns.
Matrix-valued vertical chains must keep participating K/N dimensions at most
128, including the final feature width. Large row counts are allowed until a
later contraction actually consumes that row axis.

For `MM → PW* → sum/max/mean/...`, the producer K may be up to 512 while
individual reduced axes remain at most 128. A terminal one-value-per-row
matrix contraction uses this same geometry rule; MV is one example.

Ordinary reductions track how many original elements each result represents.
Successive reductions of 64 and 64 yield an extent of 4096 and the second
reduction is excluded. A full-shaped input at a pointwise merge takes precedence
over broadcast statistics. MM replaces K with a new feature axis and resets
this coverage: a small `64 → 64 → 64` MM chain is limited by its contraction
depth and dimensions, rather than a product of its K values.

Whole-graph path reentry, device transfers and alias mutations are boundaries.
These constraints select plausible small fusion tasks; they do not certify
that a backend emits one GPU kernel.

## Rough algorithm

```text
for each supported MM seed:
    region = seed
    visit downstream values and close sibling MM branches at joins
    accept additions satisfying contraction geometry and size budgets
    try complete upstream prologue cones within those same budgets
    outputs = escaping values requiring MM + substantive computation
    prune nodes unnecessary for those outputs
    reject path reentry, unsafe aliases, or excessive arguments/outputs
deduplicate identical regions and redundant subsets with the same outputs
```
