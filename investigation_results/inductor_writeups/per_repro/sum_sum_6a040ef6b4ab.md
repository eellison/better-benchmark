# sum_sum_6a040ef6b4ab

## Status

- Family: `multi_output_reduction_templates`
- Closure status: `at_floor`
- Artifact: `repros/canonical/sum_sum_6a040ef6b4ab/oracle_dcgan_bn_backward.py`
- Classification: `BANDWIDTH_BOUND`

## Full-Scope Contract

The oracle computes the complete DCGAN batchnorm-backward leaky-ReLU scope from
Repro.forward, including both shared channel reductions, the dependent full
tensor epilogue, and the scale-gradient vector. It uses a split-K
multi-accumulator Triton reduction followed by exact f32 scalar finalization and
a dense epilogue writer.

- Inputs: `T([1024, 128, 16, 16], f32)` x3, `T([1, 128, 1, 1], f32)`, `T([128], f32)` x2
- Models: torchbench_dcgan_train_001
- Correctness: PASS, output0_max_diff=9.54e-07, output1_max_diff=6.10e-04

## Timings

- Oracle: 257.31 us
- torch.compile (combo+CDT): 268.03 us
- Ratio: 1.042x (effectively at floor)

## Gap Diagnosis

Inductor's measured full-scope output is already within the CUDAGraph harness
floor for the same required reads, f32 channel reductions, coefficient
finalization, and dense output store. The full computation is dominated by
mandatory dense tensor traffic and exact f32 reduction/epilogue work.
Classification: BANDWIDTH_BOUND -- no action needed.

## Validation

```bash
python repros/canonical/sum_sum_6a040ef6b4ab/oracle_dcgan_bn_backward.py --check
python repros/canonical/sum_sum_6a040ef6b4ab/oracle_dcgan_bn_backward.py --bench --warmup 10 --rep 50
```
