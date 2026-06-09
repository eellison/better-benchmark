# pointwise_a2d8f03e6a24

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_a2d8f03e6a24/oracle_fused_gelu_backward.py`
- Correctness: PASS
- Oracle: 20.29 us
- Compile (cd=True): 20.29 us
- Ratio: 1.000
- Status: AT_FLOOR

## Diagnosis

Inductor perfectly matches the oracle for this fused GELU backward pattern (two outputs [1024, 4096]). No gap to investigate.
