# sum_sum_baa6d8522274

## Status

- Family: `multi_output_reduction_templates`
- Closure status: `at_floor`
- Artifact: `repros/canonical/sum_sum_baa6d8522274/oracle_dual_half_channel_sum.py`
- Classification: `BANDWIDTH_BOUND`

## Full-Scope Contract

The oracle computes the full SqueezeNet captured scope with a two-stage split
reduction that evaluates the lower and upper channel halves together, preserving
the exact bool-to-f32, multiply-by-2, second multiply, mask-where sequence from
Repro.forward.

- Inputs: `T([512, 512, 13, 13], b8)`, `T([512, 512, 13, 13], f32)`, `T([512, 256, 13, 13], b8)` x2, scalar
- Models: torchbench_squeezenet1_1_train_001
- Correctness: PASS, output0_max_diff=1.46e-03, output1_max_diff=1.46e-03

## Timings

- Oracle: 102.24 us
- torch.compile (combo+CDT): 104.93 us
- Ratio: 1.026x (effectively at floor)

## Gap Diagnosis

The oracle and Inductor compiled path are within the floor band for this
memory-dominated pair of reductions. Both paths must read the same large bool/f32
inputs and per-output masks while producing only two tiny `[256]` outputs. The
remaining delta is not an actionable scheduler limitation. Classification:
BANDWIDTH_BOUND -- no targeted Inductor optimization indicated.

## Validation

```bash
python repros/canonical/sum_sum_baa6d8522274/oracle_dual_half_channel_sum.py --check
python repros/canonical/sum_sum_baa6d8522274/oracle_dual_half_channel_sum.py --bench --warmup 10 --rep 50
```
