# mean_sum_e5a21319f6c6


## Measured Timings
- Oracle: 282.34 us
- Compile (CDT): 534.27 us
- Ratio: 1.89x

## Classification: REDUCTION_EPILOGUE_GLOBAL_SUM

## Pattern

RMSNorm forward + global sum: bf16[1152000, 512] input, f32[512] weight.
The repro computes row-wise RMSNorm (cast to f32, pow(2), mean, rsqrt, mul weight),
casts back to bf16, then sums the entire [1152000, 512] result to a scalar bf16.

## Measurements

### Before fix

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Oracle | 282.34 | 1.000 |
| torch.compile (cd=True, combo) | 534.27 | 1.892 |
| torch.compile (cd=True, combo, mk=2) | ~721 | 2.420 |
| torch.compile (cd=True, combo, mk=3) | ~562 | 1.889 |

### After fix (split_multilayer_second_step=True)

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Oracle | 282.18 | 1.000 |
| torch.compile (cd=True, combo) | 216.74 | 0.768 |

**Result: 1.88x gap eliminated. Compiled is now faster than oracle.**

Correctness: PASS (shape=[] dtype=torch.bfloat16 max_diff=0.00e+00)

## Kernel Count

- Oracle: 3 kernels (fused RMSNorm+partial-sum, partial-reduce, final-reduce)
- Inductor before: 2 kernels (persistent RMSNorm+row-sum, then single-CTA serial reduction over 1152000 row sums)
- Inductor after: 3 kernels (persistent RMSNorm+row-sum, 141-CTA parallel partial reduction, 1-CTA final reduction of 141 values)

## Root Cause

The writeup originally blamed buffer materialization, but investigation revealed
Inductor was ALREADY fusing the row-wise sum into the RMSNorm kernel (via the
`create_multilayer_existing_ranges` path with split=-1). The intermediate is only
4.6 MB (1152000 * 4 bytes f32 row sums), not 1.15 GB.

The actual bottleneck was the **second-step reduction**: summing 1152000 row sums
to a scalar was done in a single CTA looping sequentially (triton_red with
xnumel=1, rnumel=1152000). This was because `create_multilayer_helper` creates
the second-step Reduction via the direct `Reduction()` constructor, bypassing the
split heuristic in `Reduction.create()`.

## Fix

**Commit:** `d85ab5508e3` on pr-184905

In `create_multilayer_helper` (ir.py), when the second-step reduction has:
- numel_hint <= 1 (scalar output)
- reduction numel > 8192 (large enough to benefit from parallelism)
- split_reductions enabled

Route through `Reduction.create()` (with `input_node=None` to prevent infinite
recursion) instead of the direct constructor. This enables the standard split
heuristic (141 splits for 1152000 elements on B200), turning a single-CTA serial
loop into a parallel tree reduction.

Relevant files:
- `/tmp/pytorch-work/torch/_inductor/ir.py` (line ~2142, create_multilayer_helper)
- `/tmp/pytorch-work/torch/_inductor/config.py` (split_multilayer_second_step flag)

## Status: FIX_IMPLEMENTED
