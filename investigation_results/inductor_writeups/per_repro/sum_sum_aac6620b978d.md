# sum_sum_aac6620b978d

## Compile: 16.1us, Oracle: 16.45us, Gap: 0.979x

## Diagnosis: AT_FLOOR (FULL_SCOPE)

## Root cause: Inductor already matches the full-scope oracle for this pattern. The oracle attempts to fuse operations but provides no measurable benefit over Inductor's existing schedule at this shape [1024] bfloat16 with [1024,2048] bfloat16 outputs.

## Fix path: No Inductor change needed -- performance is at the oracle floor. Mark as at_floor.

## Status: at_floor

## Details

- Model: Unknown (bfloat16 workload)
- Pattern: sum+sum reduction
- Oracle type: full_scope
- Shape: [1024] bf16 and [1024,2048] bf16 outputs
- Compile time: 16.1us vs oracle 16.45us -- effectively identical
- The oracle is marginally slower, confirming Inductor's schedule is already optimal for this workload
