# pointwise_85c62511d369

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_85c62511d369/oracle_nfnet_silu_gate.py`
- Correctness: PASS
- Oracle: 22.11 us
- Compile (cd=True): 22.53 us
- Ratio: 1.019
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this NFNet SiLU gate pattern (ratio within 2% noise). No gap to investigate.
