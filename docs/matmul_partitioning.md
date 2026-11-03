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

1. In original partition order, a **pure-pointwise partition** absorbs one
   available **consumer** `aten.mm`, `aten.addmm`, or `aten.bmm`, only through
   one of its **two multiplied operands**. For `addmm(bias, mat1, mat2)`, a
   partition feeding only `bias` does not qualify as a prologue.
2. Then a partition that still contains no MM absorbs one available **producer**
   MM of those same types.
3. An original MM and partition can each be used only once. An augmented
   partition cannot absorb another MM or partition.
4. Extract only augmented partitions containing exactly one MM. Unabsorbed
   standalone MMs are not included in these categories.

Available means not already absorbed. If a partition has multiple eligible MMs,
try them in FX order and take the first safe candidate. Check the updated
partition quotient after each absorption so successive merges cannot introduce
a cycle. Preserve all escaping outputs and reject interleaved external reads or
writes that conflict with mutations in the candidate.

Pure pointwise means every nontransparent operation carries
`torch.Tag.pointwise`. Normal BB's transparent view/metadata operations are
allowed, but gather/scatter and reductions do not make a partition pointwise.

```text
[pointwise partition] -> MM -> [after partition]

# Consumer absorption has priority:
[pointwise partition + MM] -> [after partition]
# The augmented partition is locked; it cannot also take after partition.
```

- **`prologue`**: the partition absorbed a consumer MM in step 1.
- **`epilogue`**: the partition absorbed a producer MM in step 2.

Epilogues may contain any computation admitted by normal BB, including multiple
reductions. There is no reduction-count or window-size cap
for these two categories. A `[reduction -> pointwise]` front partition is not
cut into a pointwise suffix for a prologue.

Absorption direction is recorded from the source graph; the prologue/epilogue
label is not inferred from standalone topology alone.

## Horizontal categories: existing window policy

These categories are independent of the normal-partition assignment above.
Their regions may overlap the single-MM examples.

- **`horizontal`**: 2–3 independent MMs share an operand through transparent
  operations and have equal output shapes. Each output depends on one MM only.
- **`horizontal_combine`**: the same grouping, but an operation combines the
  results of at least two MMs, such as `silu(x @ Wg) * (x @ Wu)`.

The existing group selection, local producer/consumer growth, and metadata
normalization are retained. The multiplied dimensions may differ; output shapes
must match. A group stops at three MMs.

Window limits remain in `MatmulLimits`: at most three MMs, 48 nodes, 32
compute operations, 12 inputs and four outputs. These limits apply to horizontal
windows, not whole normal partitions.

## Curation and deduplication

For prologue/epilogue, a family is the normal BB structural pattern hash. These
hashes intentionally ignore scalar literals. Curation keeps executable variants
separate using scalar-sensitive graph matching, then selects the variant with
the most source occurrences to represent each family. Only that variant's
compatible input points are exported; different literal values are not silently
replayed with the representative's code. Whole-partition views/casts are kept.

Horizontal categories use the existing metadata-proven normalization and
compute-DAG-plus-dtypes `structure_key` families. Scalar-sensitive matching,
input-spec remapping and output-alias checks preserve each chosen representative.
No count cap is applied.

Each `canonical/<category>/<mm ops>_<hash>/` contains:

- `repro.py`: executable `Repro` plus input and benchmark adapters.
- `shapes.json`: input points, model occurrences and partition metrics.
- `meta.json`: scalar-sensitive pattern hash, family and category.

The validation command reloads generated files and generates every input point.
Counts describe benchmark candidates, not proof that a backend can fuse all of
them or that fusion is faster; GPU compilation and measurement are separate.
