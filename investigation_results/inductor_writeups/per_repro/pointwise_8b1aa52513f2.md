# pointwise_8b1aa52513f2

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_8b1aa52513f2/oracle_tanh_gelu.py`
- Correctness: PASS
- Oracle: 84.19 us
- Compile (cd=True): 83.81 us
- Ratio: 0.995
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this tanh GELU pattern ([4096, 16384] output). No gap to investigate.
