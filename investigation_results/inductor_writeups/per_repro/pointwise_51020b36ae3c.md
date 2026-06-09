# pointwise_51020b36ae3c

## Classification: MULTI_OUTPUT_SHARED_REDUCTION

## Current Result

- Oracle path: `repros/canonical/pointwise_51020b36ae3c/oracle_shifted_embedding_cat.py`
- Correctness: PASS
- Oracle: 15.62 us
- Compiled (coordinate_descent_tuning=True): 19.1 us
- Ratio: 1.22x
- Status: FIXED (codegen bug resolved, residual performance gap)

## Root Cause (Historical)

Previously, the repro triggered an Inductor compiler bug in `torch/_inductor/codegen/triton_utils.py` line 168 (KeyError: 'buf0' during alignment check). This bug has been fixed -- the scheduler's buffer tracking is now consistent with the alignment checker.

The oracle produces correct `[32768, 384]` output from shifted embedding + cat pattern.

## Config Exploration

| Config | Result |
|--------|--------|
| Default (cd=True) | 19.1 us (1.22x) |

## Residual Gap (1.22x)

The remaining 1.22x gap is a performance issue. The oracle achieves 15.62 us with a more efficient fused kernel. This is now a performance investigation target rather than a crash bug.

## File References

- `torch/_inductor/codegen/triton_utils.py:168` (`_get_buffer_layout`) - previously crashed here
- `torch/_inductor/scheduler.py:10454` (`get_buffer_layout`)
