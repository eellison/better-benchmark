# sum_sum_1cea644c8da6

## Queue Position

- Family: `scheduler_fusion_pool_epilogue`
- Owner: `densenet121-bn-backward-pool`
- Oracle artifact: `repros/canonical/sum_sum_1cea644c8da6/oracle_fused_pool_epilogue.py`

## Oracle State

- Full-scope oracle consumes the same 32 inputs as `repro.py`.
- Returned outputs match `repro.py`: `mul_tensor_8 [256]` and `avg_pool2d_backward [64, 256, 28, 28]`.
- Correctness command: `python repros/canonical/sum_sum_1cea644c8da6/oracle_fused_pool_epilogue.py --check`.
- Correctness result: PASS; output 0 max abs diff `9.77e-04`, output 1 max abs diff `5.72e-06`.
- Benchmark command: `python repros/canonical/sum_sum_1cea644c8da6/oracle_fused_pool_epilogue.py --bench`.
- Measured timings: oracle `81.6 us`, compile (default) `132.8 us`.

## Gap Diagnosis

The oracle keeps Inductor's split two-output f32 reduction order but fuses the
reduction finalization and writes the pointwise epilogue directly into the
avg_pool2d_backward output. Inductor materializes the 14x14 epilogue and expands
it in a separate pool-backward kernel. The scheduler cuts across the reduction
consumers and the following layout-changing pool backward, preventing fusion.

Tested mitigations:
- `triton.multi_kernel=2`: 167.78 us (worse than baseline 132.8 us)
- `triton.multi_kernel=3`: 163.11 us (no improvement)
- `cuda.use_fast_math=True`: no measurable effect
- `triton.multi_kernel=2 + cuda.use_fast_math`: 167.81 us (no improvement)
- `triton.multi_kernel=3 + cuda.use_fast_math`: 163.11 us (no improvement)

None of the multi_kernel or fast_math configurations close the gap. The oracle's
1.63x advantage comes from fusing the dependent pool-backward expansion into the
reduction consumer, which requires a scheduler-level change.

Classification: `SCHEDULER_FUSION` -- allow the fixed 2x2 avg-pool-backward
expansion to fuse into its producer while preserving the reduction partial layout.

## Inductor Closure Path

- Implementation track: multi-output reduction fusion with dependent pool-backward epilogue.
- Candidate hook: let the scheduler recognize that the avg_pool2d_backward with kernel=[2,2] stride=[2,2] is a simple 4-way replication that can be sunk into the producer epilogue rather than requiring a separate pointwise expansion kernel.
- Guardrail: require full repro scope benchmarking; reduction-only partial measurements are not valid floors.

## Commands

```bash
python repros/canonical/sum_sum_1cea644c8da6/oracle_fused_pool_epilogue.py --check
python repros/canonical/sum_sum_1cea644c8da6/oracle_fused_pool_epilogue.py --bench --warmup 25 --rep 200
```
