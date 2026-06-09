# amax_sum_67d7c2666a5c

## Summary

- **Model**: torchbench_hf_Longformer_train_002
- **Classification**: NEW_PATTERN (Longformer sliding-window attention)
- **Oracle**: 112.58 us
- **Compile**: 237.60 us (264.67 us measured locally)
- **Ratio**: 2.11x
- **Oracle path**: `repros/canonical/amax_sum_67d7c2666a5c/oracle_longformer_full_scope.py`

## Pattern Description

This repro captures the complete Longformer sliding-window attention forward pass:
1. **Diagonal band assembly**: Transform BMM output `[72, 512, 512]` (computed in diagonal chunks) into a full attention score matrix `[2, 1024, 12, 513]` via the "skewed diagonal" trick (view -> permute -> pad -> slice_scatter chain)
2. **Edge masking**: Apply triangular masks to first/last chunks to enforce causal boundaries
3. **Key-mask bias**: Apply `-3.4e38` bias for masked-out key positions (from `arg7_1`)
4. **Softmax**: Online softmax (amax + exp + sum + div) over the 513-wide attention dimension
5. **Query zeroing**: Zero out rows where `arg8_1` is True (padding mask)
6. **Dropout**: Inductor-RNG based dropout (p=0.1, scale=1.1111)
7. **Output layout**: Pad-and-stride into the final `[96, 768, 256]` output with stride `[197120, 1, 769]`

## Root Cause Analysis

### Inductor generates 16 Triton kernels (1 extern + 15 compiled):

| Kernel | Purpose | Elements |
|--------|---------|----------|
| extern | Random seed generation | 36 |
| poi_0 | Inductor random values | 12.6M |
| poi_1 | Band assembly: last chunk -> slice | 3.15M |
| poi_2 | Band assembly: copy to middle chunks | 9.45M |
| poi_3 | Band assembly: first chunk fixup | 12.6M |
| poi_4 | Band assembly: first-chunk triangular | 12.6M |
| poi_5 | Key bias: constant fill | 2K |
| poi_6 | Key bias: ne + where | 2K |
| poi_7 | Key bias band assembly: last chunk | 262K |
| poi_8 | Key bias band assembly: middle | 1.05M |
| poi_9 | Key bias band assembly: edge fixup | 261K |
| poi_10 | Key bias band assembly: fixup 2 | 262K |
| poi_11 | Key bias band assembly: first chunk | 1.05M |
| poi_12 | Causal triangular mask (iota/le/rev/where) | 1.05M |
| red_13 | **Softmax reduction** (add + online softmax) | 24576 rows x 513 |
| poi_14 | Epilogue: exp/div/dropout/layout | 12.6M |
| poi_15 | Epilogue: zero-pad for output | 6.3M |

### The oracle uses 1 kernel:

One row-parallel kernel (24,576 rows) that for each row:
1. Computes the diagonal index into the BMM tensor on-the-fly (no materialized intermediate)
2. Applies key mask from `arg7_1`
3. Computes softmax inline
4. Applies query mask and dropout
5. Writes directly to the strided output layout

### Memory traffic comparison:

- **Inductor**: ~890 MB total memory traffic (multiple materializations of 12.6M-element intermediates)
- **Oracle**: ~151 MB total memory traffic (read input 75.5 MB + write output 75.5 MB)
- **Ratio**: 5.9x more memory traffic in Inductor

### Why Inductor cannot fuse these kernels:

1. **Shape mismatch across fusion boundary**: The assembly chain produces intermediates of different shapes (3.15M -> 9.45M -> 12.6M) that cannot be fused as pointwise kernels
2. **Scatter barriers**: `slice_scatter` and `select_scatter` operations require the destination tensor to be fully materialized before any consumer can read from it
3. **No pattern recognition**: Inductor's scheduler does not recognize that the entire view/permute/pad/slice_scatter chain is equivalent to "read diagonal elements from BMM with computed indices"

## Config Experiments

| Config | Time (us) | vs Baseline |
|--------|-----------|-------------|
| Default | 264.67 | 1.00x |
| coordinate_descent_tuning=True | 243.08 | 0.92x |
| max_autotune=True | 264.29 | 1.00x |
| split_cat_fx_passes=True | 263.79 | 1.00x |
| aggressive_fusion=True | 380.85 | 1.44x (worse) |
| realize_opcount_threshold=999 | 333.33 | 1.26x (worse) |
| realize_reads_threshold=2 | 322.44 | 1.22x (worse) |

**Conclusion**: No config adjustment helps. The gap is structural -- it requires pattern-level recognition.

## Fix Classification: NEW_PATTERN

### Required FX pass:

A **Longformer sliding-window attention canonicalization pass** that:

1. **Detects** the characteristic pattern:
   - `view(bmm, [BH, chunks, chunk_size, 1, chunk_size])` -> `permute([0,1,2,4,3])` -> `view` -> `constant_pad_nd` -> `view` (the skewed diagonal construction)
   - Followed by chains of `slice_scatter`/`select_scatter` (chunk assembly)
   - Feeding into `amax`/`sum` reduction (softmax)

2. **Rewrites** the entire subgraph into a single fused kernel that:
   - Computes diagonal BMM indices on-the-fly per output element
   - Fuses masking, softmax, dropout, and output layout in one pass
   - Eliminates all intermediate materializations

### Alternative (less invasive) approach:

A **diagonal-chunk assembly simplification** that:
- Recognizes the pad/view/slice_scatter pattern as computing a fixed index mapping
- Replaces the materialized scatter chain with a single pointwise kernel using computed indices
- This alone would reduce the 12 assembly kernels to 1-2, and the scheduler could then potentially fuse with the reduction

## Corpus Impact

### Longformer family size: 87 repros total

| Category | Count |
|----------|-------|
| pointwise | 34 |
| var_mean | 19 |
| sum | 14 |
| sum_sum_sum | 6 |
| amax_sum | 6 |
| sum_sum | 4 |
| amax_sum_sum | 3 |
| any | 1 |

### Models affected:
- `hf_AllenaiLongformerBase` (train + infer, multiple graph fragments)
- `torchbench_hf_Longformer` (train + infer, multiple graph fragments)
- Some fragments also appear in Bart, Albert, Bert models (shared code paths)

### Sibling repros with identical root cause:
- `amax_sum_68fe981b18dd` (AllenaiLongformerBase infer, 235 ops, 665us compile)
- `amax_sum_4c524f75213e` (AllenaiLongformerBase train, 187 ops, 867us compile)
- `amax_sum_9940b361e5b4` (hf_Longformer infer, 236 ops)
- `amax_sum_87e1fb077f24` (AllenaiLongformerBase train, 81 ops, smaller fragment)
- `amax_sum_93cb2fd0355b` (hf_Longformer train, 81 ops, smaller fragment)

All share the same underlying issue: Longformer's diagonal chunk attention assembly creates a chain of scatter operations that Inductor materializes individually instead of computing via index math.

## Priority Assessment

**HIGH PRIORITY** for the following reasons:
1. 87 repros affected (largest single-model family in the corpus)
2. 2.11x gap on THIS repro, up to 6.1x on larger variants (amax_sum_4c524f75213e)
3. The fix is well-understood: the oracle proves a single-kernel approach works
4. Pattern is stable and model-specific (Longformer/AllenaiLongformer), so regression risk is low with proper gating
5. The 34 "pointwise" Longformer repros are likely sub-fragments of the same assembly chain, meaning a single fix could close many repros simultaneously

## Done Criteria

1. FX pass that detects Longformer diagonal chunk assembly pattern
2. Rewrites to single fused kernel (or at minimum, collapses scatter chain to computed-index pointwise)
3. Validates on all 6 amax_sum Longformer variants
4. No regression on non-Longformer models (gate on pattern detection)
5. Achieves within 1.3x of oracle (target: <150us for this repro)
