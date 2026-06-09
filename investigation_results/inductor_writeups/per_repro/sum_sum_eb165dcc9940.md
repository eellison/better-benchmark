# sum_sum_eb165dcc9940

## Status

- Family: `multi_output_reduction_templates`
- Closure status: `good`
- Artifact: `repros/canonical/sum_sum_eb165dcc9940/oracle_densenet_pool_bn_backward.py`
- Classification: `SCHEDULER_FUSION`

## Full-Scope Contract

The oracle computes the complete DenseNet backward tail returned by
Repro.forward, sharing the sibling `sum(where)` and `sum(where * centered)`
channel reductions, preserving the sequential 16-input sliced accumulation, and
writing the dependent batch-norm epilogue directly into the fixed 2x2
avg_pool2d_backward output.

- Inputs: 16 sliced tensors (`T([64, 1024..512, 7, 7], f32)`), scalar, `T([64, 512, 7, 7], f32)` x2, `T([1, 512, 1, 1], f32)`, `T([512], f32)` x2, `T([64, 512, 14, 14], f32)`
- Models: torchbench_densenet121_train_001
- Correctness: PASS, output0_max_diff=2.44e-04, output1_max_diff=9.54e-07

## Timings

- Oracle: 57.57 us
- torch.compile (combo+CDT): 69.73 us
- Ratio: 1.211x (oracle is 21.1% faster -- valid gap)

## Gap Diagnosis

Inductor currently handles the sliced residual producer, paired reductions,
finalized scalar epilogue, and pool-backward expansion as separable generic
regions with extra intermediate traffic. The scheduler does not form one
full-scope multi-output reduction plan whose finalized channel scalars feed a
layout-changing pool-backward consumer.

The fix is SCHEDULER_FUSION: teach reduction scheduling to keep compatible
sibling channel reductions, the sliced residual producer, and the structured 2x2
pool-backward epilogue in one fused template.

## Inductor Closure Path

- Implementation track: multi-output reduction fusion with dependent pool-backward epilogue.
- Candidate hook: same as sum_sum_86cb93483e23 but at 7x7 spatial resolution with 16 sliced inputs.

## Validation

```bash
python repros/canonical/sum_sum_eb165dcc9940/oracle_densenet_pool_bn_backward.py --check
python repros/canonical/sum_sum_eb165dcc9940/oracle_densenet_pool_bn_backward.py --bench --warmup 10 --rep 50
```
