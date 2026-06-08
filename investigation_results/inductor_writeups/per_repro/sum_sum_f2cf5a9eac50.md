# sum_sum_f2cf5a9eac50

## Status

- Family: `multi_output_reduction_templates`
- Closure status: `at_floor`
- Artifact: `repros/canonical/sum_sum_f2cf5a9eac50/oracle_high_slice_bn_backward.py`
- Classification: `BANDWIDTH_BOUND`

## Full-Scope Contract

The oracle computes the complete GhostNet BN-backward-style captured scope,
including the high-channel slice of the `getitem_252 + getitem_267` producer,
both per-channel reductions, the dependent full-tensor epilogue, and the
8-element scale-gradient side output, while sharing the final reduction over the
partial chunks for `sum(slice)` and `sum(slice * centered)` in one Triton
finalizer.

- Inputs: `T([512, 16, 112, 112], f32)` x2, `T([512, 8, 112, 112], f32)`, `T([1, 8, 1, 1], f32)`, `T([8], f32)` x2
- Models: timm_ghostnet_100_train (3 variants)
- Correctness: PASS, output0_max_diff=9.54e-07, output1_max_diff=1.95e-03

## Timings

- Oracle: 399.23 us
- torch.compile (combo+CDT): 404.58 us
- Ratio: 1.013x (effectively at floor)

## Gap Diagnosis

Inductor already schedules this as a memory-traffic-dominated multi-output
reduction plus epilogue. The oracle's structural difference is only direct
high-slice producer recomputation and co-finalizing sibling summaries rather than
exposing a materially cheaper algorithm. Classification: BANDWIDTH_BOUND -- at
floor unless broader launch-overhead or memory-traffic improvements move both
Inductor and the oracle together.

## Validation

```bash
python repros/canonical/sum_sum_f2cf5a9eac50/oracle_high_slice_bn_backward.py --check
python repros/canonical/sum_sum_f2cf5a9eac50/oracle_high_slice_bn_backward.py --bench --warmup 10 --rep 50
```
