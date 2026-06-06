# mean_sum_e5a21319f6c6

## Classification: REDUCTION_EPILOGUE_GLOBAL_SUM

## Pattern

RMSNorm forward + global sum: bf16[1152000, 512] input, f32[512] weight.
The repro computes row-wise RMSNorm (cast to f32, pow(2), mean, rsqrt, mul weight),
casts back to bf16, then sums the entire [1152000, 512] result to a scalar bf16.

## Measurements

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Oracle | 297.92 | 1.000 |
| torch.compile (cd=True, combo) | 561.09 | 1.883 |
| torch.compile (cd=True, combo, mk=2) | 721.03 | 2.420 |
| torch.compile (cd=True, combo, mk=3) | 562.51 | 1.889 |

Best compile config: default (cd=True, combo_kernels=True) at 561.09 us.
Ratio: **1.883x** - significant gap.

Correctness: PASS (shape=[] dtype=torch.bfloat16 max_diff=0.00e+00)

## Kernel Count

- Oracle: 3 kernels (fused RMSNorm+partial-sum, partial-reduce, final-reduce)
- Inductor: 2 kernels (persistent RMSNorm writing [1152000,512] bf16, then global reduction over 1152000*512 elements)

## Root Cause

The oracle fuses the global sum INTO the RMSNorm kernel: each row kernel computes
the RMSNorm affine output, casts to bf16, and accumulates a per-CTA partial sum
(eliminating the 1152000*512*2 = 1.15 GB intermediate bf16 write + read). A tiny
2-level tree reduction (1152000 partials -> 1125 -> 1 scalar) finishes the sum.

Inductor cannot do this because:
1. The row-wise RMSNorm template produces a [1152000, 512] output
2. The downstream `sum.default` is a separate whole-tensor reduction
3. The scheduler materializes the intermediate because the consumer has a different
   reduction dimension (global vs row-wise)

The memory traffic difference:
- Oracle: reads x (1.15 GB) + weight (2 KB), writes partials (4.5 MB) = ~1.15 GB
- Inductor: reads x (1.15 GB) + weight, writes intermediate (1.15 GB bf16),
  then reads intermediate (1.15 GB), writes scalar = ~3.45 GB

This is a ~3x memory traffic ratio, matching the ~1.9x perf ratio (partially
hidden by compute).

## Fix Direction

The scheduler needs to recognize that a global reduction over the output of a
row-wise reduction can be partially accumulated within the row kernel. This is a
"reduction epilogue sink" optimization: the row-wise kernel should be able to
compute its output row, accumulate a partial sum per CTA, and skip materializing
the full intermediate.

Relevant files:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (fusion decisions)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (realize decisions for reduction outputs)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (codegen for partial accumulators)

This pattern affects any "row-norm then global-reduce" workload (RMSNorm+loss,
LayerNorm+loss, etc.).

## Status: DESIGN_DOC - requires scheduler enhancement for reduction-over-reduction fusion
