# sum_sum_9e5f39617df9

## Status

- Family: `multi_output_reduction_templates`
- Closure status: `at_floor`
- Artifact: `repros/canonical/sum_sum_9e5f39617df9/oracle_bn_affine_channel_sum.py`
- Classification: `BANDWIDTH_BOUND`

## Full-Scope Contract

The oracle computes the complete GhostNet f32 BatchNorm-affine,
upstream-gradient product, spatial sum, gate-mask/scale/full fallback, and final
channel sum returned by Repro.forward, atomically accumulating each `(N,C)`
spatial contribution directly into the `[72]` channel vector instead of
materializing the intermediate `[512, 72, 1, 1]` row result.

- Inputs: `T([512, 72, 28, 28], f32)`, `T([1, 72, 1, 1], f32)` x2, `T([72], f32)` x2, `T([512, 72, 28, 28], f32)`, `T([512, 72, 1, 1], f32)`, scalar
- Models: timm_ghostnet_100_train (3 variants)
- Correctness: PASS, max_diff=2.44e-04

## Timings

- Oracle: 76.51 us
- torch.compile (combo+CDT): 74.02 us
- Ratio: 0.967x (compile is faster; oracle is not a valid floor)

## Gap Diagnosis

The oracle is slower than the compiled code, indicating that Inductor's existing
reduction scheduling is already optimal for this pattern. Both paths must read
the same dense NCHW inputs, do the same f32 pointwise math, and perform the same
channel reduction. Classification: BANDWIDTH_BOUND -- at floor, no action needed.

## Validation

```bash
python repros/canonical/sum_sum_9e5f39617df9/oracle_bn_affine_channel_sum.py --check
python repros/canonical/sum_sum_9e5f39617df9/oracle_bn_affine_channel_sum.py --bench --warmup 10 --rep 50
```
