# Matmul pattern corpus (`mm_repro/`)

The corpus contains four categories: `prologue`, `epilogue`, `horizontal`, and
`horizontal_combine`. Each repro is extracted from a saved
post-grad FX model graph and includes its input configurations.

## Regenerate and validate

From the repository root, with a compatible CUDA-enabled PyTorch environment:

```bash
python scripts/extract_matmul_corpus.py --workers 8
python scripts/validate_matmul_repros.py
pytest tests/test_matmul_partitioner.py tests/test_mm_*.py
```

Extraction and validation use CPU workers with FakeTensors. They do not compile
or benchmark GPU kernels. The extractor enumerates tracked
`repros/models/**/full_graph_*.py` files, loads their Python modules and structured
sidecars, and walks the real FX graphs; graph renderings are not parsed.

The finished corpus atomically replaces `mm_repro/`. Its only top-level outputs
are `canonical/`, `report.json`, and `README.md`. Temporary captures are removed
when generation finishes. Re-running with the same sources and policy is
deterministic. Use `--output DIR`, `--workers N`, or repeat `--source FILE` for a
smaller run. `report.json` records the counts and policy fingerprints.

## Prologue and epilogue: whole normal BB partitions

Start with `capture_hook.get_fusion_partitions(gm)` unchanged. Treat each
partition as atomic: do not split a reduction→pointwise partition or grow a
window of individual operations around an MM.

1. In original partition order, a partition absorbs one available **producer**
   `aten.mm`, `aten.addmm`, or `aten.bmm`.
2. Then, a still-unused partition with **no scan and at most one reduction**
   absorbs one available **consumer** MM of those same types, only through one
   of its **two multiplied operands**. A reduction must normalize rows of the
   MM's A operand along K (see below). For `addmm(bias, mat1, mat2)`, a
   partition feeding only `bias` does not qualify as a prologue.
3. An original MM and partition can each be used only once. An augmented
   partition cannot absorb another MM or partition.
4. Extract only augmented partitions containing exactly one MM. Unabsorbed
   standalone MMs are not included in these categories.
5. Trim each augmented partition to the outputs that depend on its MM (the MM
   itself or its descendants) and the nodes they need. Values that do not
   depend on the MM, such as a mask that is also a model output or an operand
   intermediate saved for backward, are not MM fusion.
6. Keep the partition only if `0 < non-MM compute <= max_normal_compute` (31)
   after trimming, counting real computation (`is_computation`; views, casts
   and other scaffolding are free). 0 is a bare MM; larger partitions are
   mostly work around an incidental MM.
7. Keep an epilogue only if it reads at most `max_epilogue_extra_inputs` (4)
   extra tensors: inputs other than the MM operands (A, B and addmm's bias,
   followed back through views/casts) and the dropout seed buffer. A residual
   plus LayerNorm's weight and bias is 3. Prologues have no input bound: their
   A operand is computed inside the partition, so embedding tables and ids
   would count. These are the only size bounds on these two categories; the
   other `MatmulLimits` fields bound horizontal windows.

Available means not already absorbed. If a partition has multiple eligible MMs,
try them in FX order and take the first safe candidate. Check the updated
partition quotient after each absorption so successive merges cannot introduce
a cycle. Preserve all escaping outputs and reject interleaved external reads or
writes that conflict with mutations in the candidate.

A prologue partition may contain one reduction, and only if it normalizes rows
of A along K: it reduces the trailing dims of an operand-sized tensor in runs
that tile K (the whole row, or per-head groups), and reaches A with no permute
in between. That is RMSNorm/LayerNorm before a linear, per-head q/k norm, or a
K-block quant amax. Kernels fuse it by reducing each loaded A row or head
(when a tile holds the run) or by collecting the row statistics alongside the
mainloop and applying the scale after the MM,
`(x / rms) @ W = (x @ W) / rms` (DeepGEMM `tf32_hc_prenorm_gemm`). Column
statistics (over M), B-operand norms, two reductions and scans are not
prologues.

Reductions and scans are the nodes satisfying
`matmul_partitioner.is_reduction_or_scan`: no op where one output element
combines a whole axis, window or segment of an input. That covers reductions
and norms (`sum`, `var_mean`, `native_layer_norm`, `prims.sum`), softmax,
scans and prefix-sum compaction (`cumsum`, `nonzero`, `unique`), sorts
(`sort`, `topk`, `median`), pooling, losses, and scatter-accumulation
(`index_add`, `scatter_add`, `index_put(accumulate=True)`, upsample and
embedding backwards). The list is `REDUCTION_SCAN_NAMES`, plus any overload
tagged `torch.Tag.reduction` and not tagged `torch.Tag.pointwise`.

Tags alone are not enough in either direction. `Tag.reduction` is missing on
`aten.cumsum`, `aten._softmax`, `aten.native_layer_norm` and `prims.sum`.
`Tag.pointwise` is missing on gathers (`aten.embedding`, `aten.index.Tensor`,
`aten.gather`), constants (`aten.full`, `prims.iota`), views (`aten.split`,
`aten._unsafe_view`) and copies (`aten.cat`, `aten.slice_scatter`), which
Inductor lowers to Pointwise IR, so `embedding -> mm` is a prologue.

```text
[before partition] -> MM -> [after partition]

# Producer absorption has priority:
[before partition] -> [MM + after partition]
# The augmented partition is locked. The before partition stays available as a
# prologue for another consumer MM.
```

Epilogue fusion is the more common case, so a partition between two
MMs goes to the producer. `relu(x @ w1) @ w2` yields `epilogue [mm, relu]`;
`mm_1` gets a prologue only from another unused eligible partition, e.g.
`relu(x @ w1) @ sin(w2)` also yields `prologue [sin, mm_1]`.

- **`prologue`**: the partition absorbed a consumer MM in step 2.
- **`epilogue`**: the partition absorbed a producer MM in step 1.

Epilogues may contain any computation admitted by normal BB, including multiple
reductions; only the compute bound of step 6 applies. A front partition with
another kind of reduction is not cut into a pointwise suffix for a prologue.

Absorption direction is recorded from the source graph; the prologue/epilogue
label is not inferred from standalone topology alone.

## Horizontal categories: shared-operand windows

These categories are independent of the normal-partition assignment above.
Their regions may overlap the single-MM examples.

- **`horizontal`**: 2–3 independent MMs share an operand and have equal output
  shapes. Each output depends on one MM only.
- **`horizontal_combine`**: the same grouping, but an operation combines the
  results of at least two MMs, such as `silu(x @ Wg) * (x @ Wu)`.

Sharing an operand means the same slot and the same values, so the MMs
concatenate into one GEMM: `x @ a, x @ b` share the left operand (concatenate
`[a b]` along N) and `a @ x, b @ x` share the right one (concatenate along M).
`matmul_partitioner.operand_keys` gives each multiplied operand a key
`(slot, root, steps)`: walking up from the operand, copies (`clone`,
`contiguous`, `alias`, `detach`) are dropped, and casts, reshapes, expands and
permutes are recorded as steps until another op (the root) is reached. Runs of
reshapes keep the last shape and runs of permutes compose. MMs group when they
have an equal key:

| MMs | Keys | Group |
|---|---|---|
| `x @ a`, `x @ b` | `(0, x, ())` both | yes |
| `x.half() @ a`, `x.half() @ b` (two cast nodes) | `(0, x, (cast f16,))` both | yes |
| `x @ a`, `b @ x` | `(0, x, ())`, `(1, x, ())` | no |
| `x @ a`, `x.t() @ b` | `(0, x, ())`, `(0, x, (permute (1, 0),))` | no |
| `x.half() @ a`, `x.bfloat16() @ b` | `(0, x, (cast f16,))`, `(0, x, (cast bf16,))` | no |

The window contains the operand paths up to where they meet, even when they
are computed from an earlier MM, which local producer growth otherwise leaves
out. Without them, `x -> view_1 -> mm_1` and `x -> view_2 -> mm_2` would
extract as two MMs reading separate inputs `view_1` and `view_2`; one view
feeding both MMs is already a single input and stays out. A window that cannot
keep the paths within its limits is dropped, and validation checks that every
horizontal repro's MMs share an operand by the same key.

The existing group selection, local producer/consumer growth, and metadata
normalization are retained. The multiplied dimensions may differ; output shapes
must match. A group stops at three MMs.

Window limits remain in `MatmulLimits`: at most three MMs, 48 nodes, 32
compute operations, 12 inputs and four outputs. These limits apply to horizontal
windows, not whole normal partitions.

## Curation and deduplication

Every repro's pattern hash, its directory name, is the BB `compute_dag_signature`
without scalar literals (`pattern_hash_for_subgraph`, `dag_v1` in meta.json),
exactly as for repros/: scalar literals (a scale or eps, numeric kwargs),
device literals (`cuda:0` vs `cuda:1`), the order of
independent nodes and the output order do not change it. Which result of a
tuple op is read, and literals that change shapes such as slice bounds (SymInt
arguments), still do.

Partitions with one pattern hash are one repro: as in repros/, the first
captured partition's code is kept and every input shape becomes a shape point
of it, so literal variants run with that code's literals. Curation matches
partitions with one pattern hash node by node in its canonical order
(`canonical_node_order`). For prologue/epilogue, the family is the same pattern
hash. Whole-partition views/casts are kept, but every category reads each
`inductor_lookup_seed` from slot 0: the slot (which dropout of the graph) is a
bare-int argument the hash keeps, yet it does not change the kernel.

Horizontal categories use the existing metadata-proven normalization and
compute-DAG-plus-dtypes `structure_key` families. Scalar-sensitive matching,
input-spec remapping and output-alias checks preserve each chosen representative.
No count cap is applied.

Each `canonical/<category>/<mm ops>_<hash>/` contains:

- `repro.py`: executable `Repro` plus input and benchmark adapters.
- `shapes.json`: input points, model occurrences and partition metrics.
- `meta.json`: pattern hash, family and category.

The validation command reloads generated files and generates every input point.
Counts describe benchmark candidates, not proof that a backend can fuse all of
them or that fusion is faster; GPU compilation and measurement are separate.
