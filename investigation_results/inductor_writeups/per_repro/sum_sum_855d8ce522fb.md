# sum_sum_855d8ce522fb

## Status

- Family: `unclassified_priority_gap`
- Closure status: `at_floor`
- Artifact: `repros/canonical/sum_sum_855d8ce522fb/oracle_split_batch_hardswish_sum.py`
- Classification: `BANDWIDTH_BOUND`

## Full-Scope Contract

The oracle computes the complete MobileNetV3 BatchNorm-affine, hard-swish,
upstream-gradient product, spatial sum, gate-mask/scale/full fallback, and final
channel sum returned by Repro.forward. It uses a split batch/channel reduction
where the first Triton kernel emits the exact post-spatial partial for each
`(N,C)` row and a second kernel finalizes the `[480]` channel vector.

- Inputs: `T([512, 480, 14, 14], f32)`, `T([1, 480, 1, 1], f32)` x2, `T([480], f32)` x2, `T([512, 480, 14, 14], f32)`, `T([512, 480, 1, 1], f32)`, scalar
- Models: timm_mobilenetv3_large_100_train (4 variants), torchbench_mobilenet_v3_large_train_001
- Correctness: PASS, max_diff=1.37e-04

## Timings

- Oracle: 118.46 us
- torch.compile (combo+CDT): 121.70 us
- Ratio: 1.027x (effectively at floor)

## Gap Diagnosis

The oracle and compiled code are at the same performance floor (ratio 1.03x).
Inductor already lowers the decomposed broadcast pointwise hard-swish producer
and the dependent spatial-plus-batch reductions through generic reduction
scheduling. The full-scope Triton floor is only marginally faster than compiled
code. Classification: BANDWIDTH_BOUND -- no action needed.

## Validation

```bash
python repros/canonical/sum_sum_855d8ce522fb/oracle_split_batch_hardswish_sum.py --check
python repros/canonical/sum_sum_855d8ce522fb/oracle_split_batch_hardswish_sum.py --bench --warmup 10 --rep 50
```
