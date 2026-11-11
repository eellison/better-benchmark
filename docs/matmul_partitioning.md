# Matmul pattern corpus (`mm_repro/`)

`mm_repro/` holds benchmark repros of matmul fusion patterns in five
categories: `prologue`, `epilogue`, `grouped`, `horizontal` and
`horizontal_combine`. Each
repro is a region of a saved post-grad FX model graph (`repros/models/**/
full_graph_*.py`) with every input configuration (shape point) it was seen
with. The layout and input generation are those of `repros/`, so the repro
tooling runs on it unchanged.

## Regenerate and validate

From the repository root:

```bash
python scripts/extract_matmul_corpus.py --workers 32              # regenerate mm_repro/
CUDA_VISIBLE_DEVICES=0 python scripts/validate_matmul_repros.py   # eager GPU run of every point
pytest tests/test_matmul_partitioner.py tests/test_mm_*.py
```

Extraction runs on CPU workers with FakeTensors and needs no GPU. It
enumerates the tracked `repros/models/**/full_graph_*.py` files, loads their
Python modules and structured input sidecars, and walks the real FX graphs;
graph renderings are not parsed. The finished corpus atomically replaces
`mm_repro/`, whose only top-level outputs are `canonical/`, `report.json` and
`README.md`. Re-running with the same sources and policy is deterministic.
`--output DIR`, `--workers N` and repeated `--source FILE` make smaller runs.
`report.json` records the counts, the size limits and policy fingerprints.

Validation runs every repro eagerly on a GPU, each in its own subprocess so a
device assert cannot poison the sweep. For every shape point it generates the
inputs with `repro_harness`, runs `Repro` on them, then traces a second run
with fresh inputs in `make_fx` real mode (also on the GPU, no FakeTensor) and
checks that the graph keeps its category, its single MM
(prologue/epilogue/grouped) and its pattern hash. Non-finite outputs are reported but do not fail the run
(see Known limitations).

## Existing repro tooling

`mm_repro/canonical/<category>_<mm ops>_<hash>/` is flat like
`repros/canonical/<id>/`, and every `repro.py` uses `repro_harness` for inputs
and benchmarking, so the standard tools take the MM corpus as their corpus
root or canonical directory:

```bash
python scripts/validate_corpus_invariants.py --corpus-root mm_repro
python scripts/validate_eager.py mm_repro/canonical --all-shapes --gpus 0,1 --max-workers 4
python scripts/bench_parallel.py mm_repro/canonical --all-shapes --gpus 0,1 --output mm_results.json
python mm_repro/canonical/<id>/repro.py --all-shapes
```

A benchmark set (`benchmarks/*.json`) names repro directories relative to
`--canonical-dir`, so `scripts/select_benchmark_set.py --canonical-dir
mm_repro/canonical` builds one for this corpus and `--benchmark-set ...
--canonical-dir mm_repro/canonical` runs it. MM repros have no `oracle.py`, so
oracle sweeps skip them; `oracle_harness.load_repro_module` loads them like
any canonical repro.

## How a region becomes a repro

1. **Load**: run a source graph's module under FakeTensors and retrace it.
2. **Select regions** (`matmul_partitioner.get_matmul_regions`):
   prologue/epilogue/grouped from whole normal BB partitions, horizontal windows around
   MMs that share an operand. Both are described below. A region returns
   values, not views of them (see Outputs).
3. **Capture** (`capture_hook`): copy each region into a standalone
   GraphModule. Values from outside become inputs whose shape, stride, storage
   offset, dtype, device and aliasing are recorded; shape literals become
   `_shape_param` inputs. The pattern hash ignores shapes and the shape hash
   identifies the input specs; a repeated (pattern, shape) pair only adds an
   occurrence.
4. **Curate** (`scripts/curate_mm_corpus.py`): merge captures by pattern hash
   and export one repro per hash with all of its shape points.
5. **Validate**: the eager GPU run above.

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
   itself or its descendants), returned as values rather than views of them
   (see Outputs), and the nodes they need. Values that do not depend on the
   MM, such as a mask that is also a model output or an operand intermediate
   saved for backward, are not MM fusion.
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
8. Drop a region that reads a CPU tensor while its MM runs on the GPU
   (`cpu_inputs`). DeBERTa-v2 divides attention scores by a 0-d CPU tensor,
   `div(permute_1, arg3_1)` with `arg3_1: bf16[]` on the CPU; a GPU benchmark
   of that region also measures getting a host value into the kernel.
   Constants a region builds itself, such as `full([], 8.0, device=cpu)`, are
   not inputs and stay. A horizontal window with a CPU input falls back to a
   smaller window without it.

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
`matmul_partitioner.is_reduction_or_scan`: ops where one output element
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

Epilogue fusion is the more common case, so a partition between two MMs goes
to the producer. `relu(x @ w1) @ w2` yields `epilogue [mm, relu]`; `mm_1` gets a
prologue only from another unused eligible partition, e.g.
`relu(x @ w1) @ sin(w2)` also yields `prologue [sin, mm_1]`. Mid-layer
pre-norms therefore usually stay in epilogues: the residual + norm partition
takes the previous layer's output projection first.

- **`prologue`**: the partition absorbed a consumer MM in step 2.
- **`epilogue`**: the partition absorbed a producer MM in step 1.

Epilogues may contain any computation admitted by normal BB, including multiple
reductions; only the bounds of steps 6 and 7 apply. A front partition with
another kind of reduction is not cut into a pointwise suffix for a prologue.
The prologue/epilogue label is the absorption direction in the source graph,
not inferred from standalone topology.

## Grouped: whole partitions around `aten._grouped_mm`

`mm_grouped_absorption` gives every `aten._grouped_mm` the whole normal BB
partitions that produce its two multiplied operands (prologue side, no
reduction/scan restriction) and that consume its result (epilogue side),
through unpartitioned views. Partitions are not consumed, so back-to-back
grouped MMs share the partition between them; mm/addmm/bmm absorption is
separate and unchanged. `offs` (and a bias) is not multiplied: its producer
stays outside and `offs` is an input. In gpt-oss the routing partition (topk,
sort, row gather, sentinel mask) feeds `offs` through `histc -> cumsum` outside
the region; that one re-entry edge is allowed (`_ReplaySafety.safe(exempt=)`).

The region is trimmed with `trim_to_matmul` and needs non-MM computation and
no CPU input, but `max_normal_compute` and `max_epilogue_extra_inputs` do not
apply: a grouped region is a whole MoE block. gpt-oss's down-projection region
has 32 non-MM ops and its epilogue side reads 6 extra tensors (expert bias,
router scores, permutation, sentinel mask, residual, norm weight); both counts
are recorded in the partition metrics.

Curation annotates every placeholder used as `offs` with
`{"kind": "offsets", "high": G + 1, "end": G}`, G the grouped dim (rows of A
for 2d x 3d): `repro_harness` draws sorted values in `[0, G]` and sets the last
to G. Random ints device-assert (an offset past G) or leave rows unwritten
(a last offset short of G).

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

The multiplied dimensions may differ; output shapes must match. A group stops
at three MMs. Its window starts from the MMs and the work that prepares their
operands, grows forward to consumers and backward to side inputs, and keeps the
largest window within `MatmulLimits`: at most three MMs, 48 nodes, 32 compute
operations, 12 inputs and four outputs, counted after outputs become values
(see Outputs).

The window contains the operand paths up to where they meet, even when they
are computed from an earlier MM, which local producer growth otherwise leaves
out. Without them, `x -> view_1 -> mm_1` and `x -> view_2 -> mm_2` would
extract as two MMs reading separate inputs `view_1` and `view_2`; one view
feeding both MMs is already a single input and stays out. A window that cannot
keep the paths within its limits is dropped, and validation checks that every
horizontal repro's MMs share an operand by the same key.

## Outputs: values, not views

A view returns its input's storage, so a region that returns a view of a value
returns that value. In every category, extraction replaces each output before
capture, and the pattern hash, the inputs and the code never see the views:

1. An output that views a value whole is that value. Views that keep every
   element of their input are dropped from its end: `view`, `reshape`,
   `_unsafe_view`, `squeeze`, `unsqueeze`, `permute`, `t`, `transpose`,
   `expand`, `alias`, `detach`, and a `slice` or `select` that covers its whole
   input. Outputs that become one value are one output.
2. A view that drops elements, such as a `slice` or `select` of part of a
   tensor, stays an output unless it views another output.
3. Casts and clones write a new tensor and stay outputs.

M2M100's addmm + residual + LayerNorm epilogue, read by 24 consumers through
their own view, returned the norm and 24 views of it, each view with its own
`_shape_param` input:

```python
convert_element_type_1 = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16)
view_1 = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1)
...
view_24 = torch.ops.aten.view.default(convert_element_type_1, _shape_param_24)
return (convert_element_type_1, view_1, ..., view_24)
```

It now ends in `return convert_element_type_1`, and the same epilogue read
through any number of views has the same pattern hash. A QKV window's
`return (addmm_2, view_3, view_4, view_5)`, where `view_3`, `view_4` and
`view_5` reshape `addmm_2`, `addmm` and `addmm_1`, returns
`(addmm, addmm_1, addmm_2)`.

Views that drop elements keep the outputs a model reads. DeiT's
`return (select, select_1)` keeps its two class tokens of a
`[128, 198, 768]` norm; returning the norm would claim all 198 tokens. When the
viewed value is an output anyway, the view adds nothing:

```python
add = torch.ops.aten.add.Tensor(arg3_1, view)        # [16, 128, 768], BERT_pytorch
select = torch.ops.aten.select.int(add, 1, 0)        # [16, 768]
view_1 = torch.ops.aten.view.default(add, _shape_param_1)
return (select, view_1)                              # now: return add
```

A clone such as `clone(permute, memory_format=torch.contiguous_format)` stores
a transposed copy, so an epilogue that returns it keeps that store.

## Curation

Every repro's pattern hash, its directory name, is the BB `compute_dag_signature`
without scalar literals (`pattern_hash_for_subgraph`, `dag_v1` in meta.json),
exactly as for `repros/`, in every category. Scalar literals (a scale or eps,
numeric kwargs), device literals (`cuda:0` vs `cuda:1`), the order of
independent nodes and the output order do not change it. Which result of a
tuple op is read, and literals that change shapes such as slice bounds (SymInt
arguments), still do.

Captures with one pattern hash are one repro: as in `repros/`, the first
captured region's code is kept and every input shape becomes a shape point of
it, so literal variants run with that code's literals. Curation maps a capture
onto the repro node by node in the hash's canonical order
(`canonical_node_order`) and checks that the repro's code, replayed with the
capture's inputs, reproduces the capture's output metadata, devices and
aliases; it stops with an error otherwise. There is no further family
deduplication, and no cap on repros or shape points.

Before hashing, curation reads every `inductor_lookup_seed` from slot 0: the
slot (which dropout of the graph) is a bare-int argument the hash keeps, yet it
does not change the kernel. Prologue/epilogue partitions keep their inner
views and casts (output views are already gone). Horizontal windows first get
metadata-proven normalization (`mm_corpus_dedup.normalize_graph`: views
removed only when their metadata proves them identity, input permutations
folded into strided input specs) and must still fit `MatmulLimits`.

Each `canonical/<category>_<mm ops>_<hash>/` contains:

- `repro.py`: the `Repro` module, `make_inputs`, and `repro_harness.benchmark_repro`
  as its entry point.
- `shapes.json`: shape points (compact input specs), their model occurrences
  and partition metrics.
- `meta.json`: pattern hash and hash modes, `kind: matmul`, category and the
  number of shape points.

## Input generation

`make_inputs` calls `repro_harness.make_inputs_from_config`, the generator of
every repro in `repros/`. Each shape point records each tensor's shape,
dtype, stride, storage offset and device; inputs that share storage in the
model are one alias group, allocated once at the recorded size. Floats are
random normal, integers and bools come from `[0, 100)` and `{0, 1}` unless the
capture recorded a generator such as index bounds, and inputs that only feed
`sqrt`/`rsqrt` (an external variance) are generated positive
(`mm_input_constraints`).

## Known limitations

- A pattern can read statistics as independent random inputs, e.g. a softmax
  backward's max and sum, and then produce non-finite outputs from finite
  inputs. Validation reports such points; the current corpus has none.
- Counts describe benchmark candidates, not proof that a backend can fuse all
  of them or that fusion is faster; compilation and measurement are separate
  (`bench_parallel.py`).
