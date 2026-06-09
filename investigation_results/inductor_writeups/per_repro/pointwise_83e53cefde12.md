# pointwise_83e53cefde12

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_83e53cefde12/oracle_nfnet_gated_gelu.py`
- Correctness: PASS
- Oracle: 18.14 us
- Compile (cd=True): 16.16 us
- Ratio: 0.891 (oracle slower)
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than compiled output. Inductor outperforms this NFNet gated GELU oracle. No gap to investigate - the oracle needs updating.
