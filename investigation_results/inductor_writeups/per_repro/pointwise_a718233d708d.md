# pointwise_a718233d708d

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_a718233d708d/oracle_f32_div_fp8_transpose.py`
- Correctness: PASS
- Oracle: 35.62 us
- Compile (cd=True): 35.58 us
- Ratio: 0.999
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this f32 div + fp8 transpose pattern ([768, 50304] float8_e4m3fn output). No gap to investigate.
