# sum_sum_847b1e4f73dd

## Status

- Family: `multi_output_reduction_templates`
- Closure status: `at_floor`
- Artifact: `repros/canonical/sum_sum_847b1e4f73dd/oracle_ghostnet_bn_backward_shared_finalize.py`
- Classification: `BANDWIDTH_BOUND`

## Full-Scope Contract

The oracle computes the complete GhostNet BN-backward-style captured scope,
including the masked sliced-add producer, both per-channel reductions, the
dependent full-tensor epilogue, and the 8-element scale-gradient side output,
while sharing the final reduction over the 64 partial chunks for
`sum(where * centered)` and `sum(where)` in one Triton finalizer.

- Inputs: `T([512, 16, 112, 112], f32)`, `T([512, 8, 112, 112], f32)` x3, `T([1, 8, 1, 1], f32)`, `T([8], f32)` x2
- Models: timm_ghostnet_100_train (3 variants), torchbench_timm_vovnet_train_001
- Correctness: PASS, output0_max_diff=1.91e-06, output1_max_diff=2.50e-01

## Timings

- Oracle: 505.31 us
- torch.compile (combo+CDT): 514.91 us
- Ratio: 1.019x (effectively at floor)

## Gap Diagnosis

Inductor already reads the big input tensors once with two accumulators and the
oracle's only structural difference is avoiding one tiny sibling-finalizer
launch before the same bandwidth-heavy epilogue. The scheduler does not fuse
equal-`xnumel/rnumel` final-reduction epilogues, but bench_oracle shows that
difference is hidden by full-tensor memory traffic. Classification:
BANDWIDTH_BOUND -- no action needed.

## Validation

```bash
python repros/canonical/sum_sum_847b1e4f73dd/oracle_ghostnet_bn_backward_shared_finalize.py --check
python repros/canonical/sum_sum_847b1e4f73dd/oracle_ghostnet_bn_backward_shared_finalize.py --bench --warmup 10 --rep 50
```
