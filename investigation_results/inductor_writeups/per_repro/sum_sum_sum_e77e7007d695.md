# sum_sum_sum_e77e7007d695

## Classification: MULTI_OUTPUT_SHARED_REDUCTION

## Current Result

- Oracle path: `repros/canonical/sum_sum_sum_e77e7007d695/oracle_nfnet_gated_gelu_multi_output.py`
- Correctness: PASS
- Oracle: 191.97 us
- Compile (cd=True): 239.26 us
- Ratio: 1.246
- Best config: combo+mk=3: 239.58 us (ratio 1.251)
- Status: real_gap

## Config exploration results

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Default (cd=True) | 239.26 | 1.246 |
| combo+mk=2 | 241.63 | 1.261 |
| combo+mk=3 | 239.58 | 1.251 |
| Oracle | 191.97 | 1.000 |

No config closes the gap.

## Root Cause

The oracle computes the complete NFNet gated exact-GELU-backward reduction scope by streaming the shared sigmoid/gate/GELU-gradient pointwise producer once into per-(N,C) spatial summaries and finalizing the scalar loss-gradient sum plus the two [512] channel reductions.

Inductor schedules the fused producer, global sum, spatial sigmoid-gradient reduction, and channel reductions as separate generic regions over materialized intermediates. The scheduler does not form one multi-output reduction plan for sibling reductions with different output ranks when one output has a per-(N,C) reduced epilogue before the final channel sum.

## Kernel count
- Oracle: 2 kernels (spatial partial reduction + finalize)
- Inductor: 4+ kernels (separate reductions for each output)

## Recommendation

Requires scheduler enhancement to keep the shared pointwise producer inside a multi-accumulator spatial reduction and lower the scalar and channel finalizers together.

File: `torch/_inductor/scheduler.py` (multi-output reduction with mixed output ranks)
