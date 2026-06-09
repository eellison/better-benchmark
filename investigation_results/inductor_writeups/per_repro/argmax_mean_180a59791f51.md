# Gap Diagnosis: argmax_mean_180a59791f51 (reformer_lsh_routing)


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 287.74 us
- Ratio: N/A

## Classification: NEW_PATTERN (LSH routing idiom)

## Measurements

| Config | Time (us) | Ratio vs Default |
|--------|-----------|------------------|
| Default | 284.5 | 1.00x |
| Combo Looped (CD) | 278.4 | 1.02x |
| Combo Persistent (CD) | 257.8 | 1.10x |
| Oracle (structured eager) | ~124-144 | ~2.0-2.3x |

- **Bytes accessed**: 811.6 MB (logical)
- **Best compile SOL**: 2.08 GB/s effective BW ratio (combo_persistent)
- **Model**: torchbench_hf_Reformer_train

## Pattern Description

Reformer LSH (Locality Sensitive Hashing) routing forward pass:
1. Compute argmax(cat([x, -x])) to get 0..127 bucket assignment
2. Sort by bucket*4096 + position to get sorted indices
3. Gather from sorted indices, compute RMS scaling, cyclic concatenation
4. Produce 5 outputs: sorted_keys, bucket view, gathered tensor, normalized output,
   and a secondary gathered+concatenated output

## Root Cause

Inductor emits:
1. A pointwise kernel that writes the full cat([x, -x]) buffer (materializing a
   [8, 12, 4096, 128] tensor = 192MB when only the argmax is needed)
2. A separate argmax/key kernel
3. An external sort (not fusible)
4. Post-sort pointwise/gather/RMS kernels

The main waste is materializing the 128-wide concatenated buffer when only the argmax
position is needed. The oracle computes the same argmax using max(x) and max(-x)
(signed-abs argmax) without ever allocating the concatenated tensor.

## Oracle Strategy

1. **Signed argmax**: Instead of `argmax(cat([x, -x]))`, compute:
   - pos_val, pos_idx = x.max(dim=-1)
   - neg_val, neg_idx = (-x).max(dim=-1)
   - bucket = where(pos_val >= neg_val, pos_idx, neg_idx + 64)
   
   This produces the identical 0..127 bucket assignment without the 192MB allocation.

2. **Standard sort**: Sort bucket*4096 + position (same as Inductor)

3. **Fused gather/RMS/concat**: Apply sorted gathers, RMS scaling, and cyclic
   concatenation in a single pass

## Inductor Fix Required

**NEW_PATTERN**: A lowering pattern that canonicalizes `argmax(cat([x, -x]))` into a
signed-abs argmax (computing max and max-of-negation in parallel). Additionally:
1. Recognize that the external sort boundary prevents pulling the argmax algebra into
   consumers
2. The post-sort gather/RMS/cyclic-concat epilogue could be fused into fewer kernels

The key insight is algebraic: `argmax(cat([x, -x]))` over dimension d is equivalent to
comparing max(x) vs max(-x) and selecting the appropriate index, which halves memory
traffic for the argmax computation.

## Generalizability

LOW - This is highly specific to the Reformer LSH routing mechanism. The algebraic
optimization (signed-abs argmax) is a valid general transform for `argmax(cat([x, -x]))`
patterns, but Reformer is the only known model using this idiom. The main value is as
a pattern-matching canonicalization in the graph optimizer.

## Remaining Gap After CD

~114-134us (258us best compile vs ~124-144us oracle). Most of the remaining gap is
the materialized cat buffer and the inability to fuse across the sort boundary.
