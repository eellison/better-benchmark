# sum_sum_975c7af7df3e

## Status

- Family: `multi_output_reduction_templates`
- Closure status: `good`
- Artifact: `repros/canonical/sum_sum_975c7af7df3e/oracle_resnest_bn_backward.py`
- Classification: `SCHEDULER_FUSION`

## Full-Scope Contract

The oracle computes the complete ResNeSt BN-backward-style captured scope,
including the expanded grouped producer, ReLU mask select, both channel
reductions, the scale-gradient vector, and the full dense epilogue, while
keeping the two sibling channel reductions in one Triton pass with `sum(where)`
and `sum(where * centered)` accumulators and recomputing the cheap producer for
the epilogue instead of materializing it.

- Inputs: `T([32, 2, 1, 64], f32)`, `T([32, 2, 64, 56, 56], f32, stride=(200704, 0, ...))`, `T([32, 64, 1, 1], f32)`, `T([32, 128, 56, 56], f32)`, scalar, `T([1, 128, 1, 1], f32)` x2, `T([128], f32)`, shapes
- Models: torchbench_timm_resnest_train_001
- Correctness: PASS, output0_max_diff=1.91e-06, output1_max_diff=9.77e-04

## Timings

- Oracle: 98.69 us
- torch.compile (combo+CDT): 114.18 us
- Ratio: 1.157x (oracle is 15.7% faster -- valid gap)

## Gap Diagnosis

Inductor does not co-schedule this multi-output reduction today because the
scheduler fails to fuse sibling reductions over the same `numel/rnumel` and
shared viewed/expanded masked producer once their summaries feed different
epilogues.

The fix is SCHEDULER_FUSION: teach the reduction scheduler to recognize
shared-input sibling channel reductions with equal reduction domains and emit
one multi-accumulator reduction plus the dependent dense epilogue.

## Inductor Closure Path

- Implementation track: multi-accumulator sibling reduction fusion.
- Candidate hook: recognize that two `sum.dim_IntList` reductions share the same viewed/expanded producer and reduction domain, enabling single-pass multi-accumulator codegen.

## Validation

```bash
python repros/canonical/sum_sum_975c7af7df3e/oracle_resnest_bn_backward.py --check
python repros/canonical/sum_sum_975c7af7df3e/oracle_resnest_bn_backward.py --bench --warmup 10 --rep 50
```
