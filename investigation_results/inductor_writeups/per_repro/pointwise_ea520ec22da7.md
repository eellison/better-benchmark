# pointwise_ea520ec22da7

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_ea520ec22da7/oracle_qkv_as_strided_alias.py`
- Correctness: PASS
- Oracle: 2.78 us
- Compile (cd=True, combo=True): 2.75 us
- Ratio: 0.989 (compile is FASTER than oracle)
- Status: AT_FLOOR

## Diagnosis

Inductor matches the oracle for this QKV as_strided alias pattern on [128, 12, 197, 64] tensors. Both produce zero-kernel metadata-only results (CUDA graph empty warning confirms no GPU work). No gap to investigate.
