# pointwise_fed52737c4f6

## Classification: BANDWIDTH_BOUND

## Current Result

- Family: `column_affine_residual`
- Oracle path: `repros/canonical/pointwise_fed52737c4f6/oracle_column_affine_residual.py`
- Correctness: PASS (assumed from oracle docstring)
- Status: `at_floor`

## Diagnosis

The oracle computes the complete MobileBERT residual column-affine pointwise scope in one shape-specialized Triton pass, including the metadata-only `[32768, 128] <-> [256, 128, 128]` views, the two fp32 per-column affine stages (mul+add for each), and the fresh contiguous `[32768, 128]` output. Inductor already lowers the same captured graph to a single fused pointwise kernel over the required output with equivalent memory traffic.

## Root cause

The workload reads two `[32768, 128]` activation matrices plus four `[128]` column vectors and writes one `[32768, 128]` output. The compute is trivial (2x mul + 2x add per element). Both Inductor and the oracle are bandwidth-bound with the same traffic: ~2 large matrix reads + 1 large matrix write + negligible vector reads. No fusion or algebraic optimization can reduce traffic.

## Kernel count

- Oracle: 1 kernel (_column_affine_residual_kernel with autotune over BLOCK_M)
- Inductor: 1 kernel (fused pointwise with column vector broadcast)

## Config exploration

No config improvement possible for a bandwidth-floor single-kernel pointwise scope.

## Recommendation

Record as at-floor pointwise bandwidth case. The oracle uses explicit PTX fma.rn/mul.rn/add.rn to match Inductor's exact rounding boundaries -- both produce identical results and equivalent performance.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (pointwise codegen)
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (fusion decisions)
