# pointwise_0c71173959e2

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_0c71173959e2/oracle_relu_slice_add_mask.py`
- Correctness: PASS
- Oracle: 706.50 us
- Compile (cd=True): 704.35 us
- Ratio: 0.997
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this relu slice add mask pattern. No gap to investigate.
