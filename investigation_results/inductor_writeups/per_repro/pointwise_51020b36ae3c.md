# pointwise_51020b36ae3c

## Classification: COMPILER_BUG

## Current Result

- Oracle path: `repros/canonical/pointwise_51020b36ae3c/oracle_shifted_embedding_cat.py`
- Correctness: PASS
- Bench: FAIL (InductorError: KeyError: 'buf0')
- Status: COMPILER_BUG

## Root Cause

The repro triggers an Inductor compiler bug in `torch/_inductor/codegen/triton_utils.py` line 168:
```
layout = V.graph.scheduler.get_buffer_layout(buf_name)
         self.name_to_buf[buf_name]  -> KeyError: 'buf0'
```

This occurs during the alignment check phase (`is_unaligned_buffer`) when generating Triton code. The scheduler's `name_to_buf` dict does not contain 'buf0', suggesting a buffer was optimized away or renamed but a stale reference persists.

The oracle itself passes correctness (produces correct `[32768, 384]` output from shifted embedding + cat pattern).

## Config Exploration

| Config | Result |
|--------|--------|
| Default (cd=True) | KeyError: 'buf0' during compilation |

## Diagnosis

This is a pre-existing Inductor bug on the `pr-184905` branch. The shifted-embedding-cat pattern exercises a code path where buffer liveness analysis and the alignment checker disagree about which buffers exist.

## File References

- `torch/_inductor/codegen/triton_utils.py:168` (`_get_buffer_layout`)
- `torch/_inductor/scheduler.py:10454` (`get_buffer_layout`)
