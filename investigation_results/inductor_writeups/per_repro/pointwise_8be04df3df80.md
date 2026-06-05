# pointwise_8be04df3df80

## Classification: CAT_CHANNEL_PAIRED_STORE

## Current Result

- Family: `paired_cat_bn_relu`
- Oracle path: `repros/canonical/pointwise_8be04df3df80/oracle_paired_cat_bn_relu.py`
- Correctness: PASS
- Oracle: `46.4 us`
- `torch.compile coordinate_descent_tuning=True`: `60.32 us`
- Ratio: 1.3
- Best config: `combo_kernels=True,mk=3`: `59.97 us` (minimal improvement)
- Status: `real_gap`

## Diagnosis

The oracle computes the complete GhostNet BN-affine/ReLU plus channel-cat scope with a paired layout kernel that copies `relu_35` into the first output half while computing the normalized `convolution_84` value for the matching source coordinate into the second output half. The output shape is [512, 960, 7, 7].

Inductor lowers the cat as generic output-element layout/indexing work and cannot pair the two channel-half stores or hoist channel parameters at the row granularity. The scheduler treats cat materialization as a separate indexing concern.

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True) | 60.32 |
| combo+mk=2 | 59.48 |
| combo+mk=3 | 59.97 |
| Oracle | 46.4 |

No config closes the gap significantly.

## Root cause

The scheduler/codegen does not model fixed channel-cat materialization as a fused producer with ownership of both destination channel ranges. The oracle emits paired channel-segment stores with pointwise producers fused and per-channel BN parameters loaded once per row tile.

## Kernel count
- Oracle: 1 kernel (paired BN+ReLU+cat)
- Inductor: 2+ kernels (BN+ReLU, then cat layout copy)

## Recommendation

Fix in `torch/_inductor/scheduler.py`: teach concat/materialization scheduling to emit paired channel-segment stores with pointwise producers fused and per-channel parameters loaded once per row tile. Affected by same root cause as other `paired_cat_bn_relu` family repros.
