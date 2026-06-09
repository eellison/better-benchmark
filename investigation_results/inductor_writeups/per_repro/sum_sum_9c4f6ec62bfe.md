# sum_sum_9c4f6ec62bfe

## Classification: MULTI_OUTPUT_SHARED_REDUCTION

## Current Result

- Oracle path: `repros/canonical/sum_sum_9c4f6ec62bfe/oracle_multi_output_reduction.py`
- Correctness: PASS
- Oracle: 220.99 us
- Compile (cd=True): 278.50 us
- Ratio: 1.260
- Best config: combo+mk=2/3: 278.46-278.56 us (ratio 1.26 unchanged)
- Status: real_gap

## Config exploration results

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Default (cd=True) | 278.50 | 1.260 |
| combo+mk=2 | 278.56 | 1.260 |
| combo+mk=3 | 278.46 | 1.260 |
| Oracle | 220.99 | 1.000 |

No config closes the gap.

## Root Cause

The oracle computes the full MobileNetV3 hard-swish plus batch-norm-backward fragment from the original ten repro inputs, sharing the hard-sigmoid/average-pool gradient producer across both sum([0,2,3]) channel reductions and then reusing the finalized channel statistics to emit the returned f32[512,480,14,14] tensor and f32[480] vector.

Inductor schedules the broadcast pointwise producer, sibling reductions, and dependent BN-backward epilogues as generic reduction/pointwise work rather than one full-scope multi-output plan, so it rereads/recomputes the large producer and cannot coordinate the two accumulators and epilogues.

## Kernel count
- Oracle: 2-3 kernels (shared partial + finalize + epilogue)
- Inductor: 4+ kernels (separate reductions + pointwise)

## Recommendation

Same root cause as sum_sum_7388c7a6f044. Requires scheduler support for compatible sibling channel reductions with a shared producer, lowered as multi-output reduction with fused tensor/vector epilogues.

File: `torch/_inductor/scheduler.py`
