# sum_sum_sum_1dd09876ea9d

## Status

- Family: `multi_output_reduction_templates`
- Closure status: `good`
- Artifact: `repros/canonical/sum_sum_sum_1dd09876ea9d/oracle_nfnet_pool_silu_dual_sum.py`
- Classification: `SCHEDULER_FUSION`

## Full-Scope Contract

The oracle computes the complete NFNet pooled-SiLU backward fragment returned by
`Repro.forward`, including the fixed 2x2 avg-pool backward expansion, exact
captured sigmoid, scalar all-element reduction, per-`(N,C)` spatial reduction,
sigmoid-derivative gate, and final channel reduction.

- Inputs: `T([128, 512, 12, 12], f32)`, `T([128, 512, 24, 24], f32)` x3, `T([128, 512, 1, 1], f32)`, `T([128, 512, 24, 24], f32)`, scalar
- Models: timm_dm_nfnet_f0_train (3 variants), torchbench_timm_nfnet_train_001
- Correctness: PASS, output0_max_diff=4.88e-04, output1_max_diff=1.53e-05

## Timings

- Oracle: 145.12 us
- torch.compile (combo+CDT): 269.18 us
- Ratio: 1.855x (oracle is 85.5% faster -- large valid gap)

## Gap Diagnosis

Inductor currently treats the structured pool producer, the scalar sibling sum,
and the gated channel sum as separate generic regions even though they share the
same producer and reduction domain. The scheduler does not co-plan sibling
reductions where one output is a global scalar and the other is a channel
reduction with a per-`(N,C)` gate applied after the spatial sum.

The fix is SCHEDULER_FUSION: fuse the structured pool producer and shared
pointwise chain into one multi-output reduction template with a channel finalizer
while preserving exact f32 operation order and sigmoid numerics.

## Inductor Closure Path

- Implementation track: multi-output reduction with heterogeneous output shapes (scalar + vector).
- Candidate hook: recognize that `sum.default` (global scalar) and `sum.dim_IntList` (channel vector) share the same producer after the pool-backward expansion, enabling single-pass multi-accumulator codegen with per-accumulator finalization.
- This repro has the largest gap (1.855x) in this batch and is high priority.

## Validation

```bash
python repros/canonical/sum_sum_sum_1dd09876ea9d/oracle_nfnet_pool_silu_dual_sum.py --check
python repros/canonical/sum_sum_sum_1dd09876ea9d/oracle_nfnet_pool_silu_dual_sum.py --bench --warmup 10 --rep 50
```
