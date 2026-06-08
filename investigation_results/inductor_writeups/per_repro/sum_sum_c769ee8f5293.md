# sum_sum_c769ee8f5293

## Status

- Family: `multi_output_reduction_templates`
- Closure status: `at_floor`
- Artifact: `repros/canonical/sum_sum_c769ee8f5293/oracle_t5_dropout_dual_reduction.py`
- Classification: `BANDWIDTH_BOUND`

## Full-Scope Contract

The oracle computes the complete T5 dropout/backward scope from Repro.forward,
including the 24 matrix views plus residual add chain, the captured
`mul_tensor_2` materialization boundary, both bool-mask scales, the channel
reduction, the row reduction, and the dependent transposed dense epilogue. It
co-plans the shared producer, sibling reductions, and final transpose-view store
with Triton partial reductions plus a dense epilogue.

- Inputs: 24 `T([8192, 768], f32)` tensors, `T([8, 1024, 768], f32/b8)`, `T([768], f32)`, `T([8, 1024, 1], f32)`, shapes
- Models: torchbench_hf_T5_base_train_001
- Correctness: PASS, output0_max_diff=6.71e-04, output1_max_diff=1.53e-05

## Timings

- Oracle: 228.61 us
- torch.compile (combo+CDT): 218.72 us
- Ratio: 0.957x (compile is faster; oracle is not a valid floor)

## Gap Diagnosis

Inductor's compiled result is already within the CUDAGraph harness floor for the
same required dense reads, f32 reductions, and full output store. The full
computation is dominated by required memory traffic and exact f32
reduction/epilogue work. Classification: BANDWIDTH_BOUND -- at floor, no action
needed.

## Validation

```bash
python repros/canonical/sum_sum_c769ee8f5293/oracle_t5_dropout_dual_reduction.py --check
python repros/canonical/sum_sum_c769ee8f5293/oracle_t5_dropout_dual_reduction.py --bench --warmup 10 --rep 50
```
