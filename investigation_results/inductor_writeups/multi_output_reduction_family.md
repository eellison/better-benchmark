# Family: multi_output_reduction_templates (rank #1 by model attribution)

245 repros / 82 models / 21.8% avg fusible speedup. Investigation started 2026-06-10.

## Verified targets (B200, fresh 2026-06-10)

| repro | oracle us | compile us | ratio | kernels generated |
|-------|-----------|------------|-------|-------------------|
| sum_011e69da166d (PRIMARY) | 403.3 | 573.3 | 1.42x | **1** (fused pointwise+permute-store+colsum) |
| sum_0becf9609ad7 | 66.8 | 93.8 | 1.41x | **1** (looped INNER reduction) |
| sum_21c4018e2b7b | 44.6 | 56.0 | 1.26x | 2 (split reduction + finalize) |
| sum_031c8d6c51f7 (AT FLOOR) | ~51 | ~53 | 1.04x | 1 (persistent reduction) |

## Finding #1: the family name is misleading for these targets

The "multiple reductions re-reading the same input as separate kernels" failure mode
does NOT reproduce on the verified targets: Inductor already fuses the sibling
outputs into a single kernel for the primary (permuted side-store + column sum in
one `triton_red_fused_exp_le_mul_sum_where_0`), and 0bec/031c are single kernels.
The measured gaps come from **kernel-quality decisions inside the single fused
kernel**, not from missing fusion.

## Finding #2 (PRIMARY): coordinate-descent tuning lands in a local minimum

Hand-sweep of the EXACT inductor-generated kernel for sum_011e69da166d
(x=197951 cols contiguous, r=1024 rows, OUTER-style reduction, hint=DEFAULT):

- CD pick: `XBLOCK=32, R0_BLOCK=64, num_warps=16, stages=1` -> 576 us
- Best in same config space: `XBLOCK=128, R0_BLOCK=32, num_warps=8` -> **481 us**
- Oracle (col-tile loop over rows, BLOCK_N>=64 contiguous axis last, stages=3): 403 us

So 95us of the 170us gap is reachable *within the existing kernel* — CD never
finds it (must move XBLOCK 32->128 AND warps 16->8 simultaneously; radius-1
single-coordinate moves regress individually). The remaining ~78us appears
structural (tile orientation / pipelining).

## Status

WIP — investigating (a) why default+CD picks the bad config for DEFAULT-hint
fused reduction+side-store kernels, (b) what the at-floor sibling does
differently, (c) whether other family members show the separate-kernel
re-read pattern at all.
