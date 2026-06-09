# pointwise_1c65af29b270 - GPT-J RoPE Gather Layout

## Classification: BANDWIDTH_BOUND (minor gap)

## Benchmark Results
- Oracle: 7.07 us
- Inductor compiled: 7.52 us
- Ratio: 1.063x (oracle is 1.06x faster)

## Kernel Count
- Inductor: 1 kernel (fully fused pointwise)
- Oracle: 1 kernel (fused RoPE + layout kernel)

## Root Cause

The repro computes the GPT-J indexed RoPE (Rotary Position Embedding) for two matmul branches:
1. Position table gather: [2048,64] table indexed by [1,128] positions -> [1,128,64] coefficients
2. Split coefficients into sin/cos pairs [1,128,32] each
3. For each of mm_108 and mm_109 ([128,4096] -> [1,128,16,256]):
   - Slice first 64 dims as rotary
   - Apply rotate_half: interleave [-x_odd, x_even] pairs
   - Multiply by cos/sin coefficients and add
   - Concatenate rotary result [64] with passthrough [192]
4. Final permute to [1,16,128,256] output layout

Inductor already fuses everything into a single pointwise kernel. The oracle and Inductor both produce exactly 1 kernel. The 1.063x gap is minimal and within typical measurement noise.

The oracle kernel tiles by (rows=seq*heads, dims=head_dim) and uses explicit pair indexing for the rotate-half pattern. Inductor's kernel fuses all the views, gathers, splits, cats, and permutes through its generic index calculation machinery, which is slightly less optimal for this particular interleaved access pattern but functionally equivalent.

## Config Exploration
| Config | Time (us) | Notes |
|--------|-----------|-------|
| coordinate_descent + combo_kernels | 7.52 | Default compiled |
| Oracle (tiled RoPE kernel) | 7.07 | 1.06x faster |

## What Inductor Needs

The gap (1.063x) is at the noise floor - the oracle itself classifies this as BANDWIDTH_BOUND with no meaningful scheduler/codegen gap. Both solutions are single-kernel, fully-fused, and the difference comes from minor indexing arithmetic overhead in Inductor's generic approach vs. the oracle's specialized pair-access pattern.

No fix warranted for this magnitude of gap.

## File References
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - pointwise kernel codegen
- Oracle classification explicitly states: "no Inductor change is justified"

## Status
Closed - gap is at noise floor (1.063x). No fix warranted.
