# pointwise_e31a9c1b437e

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_e31a9c1b437e/oracle_head_layout_view.py`
- Correctness: PASS (layout check also passes: stride=(32768, 1, 64))
- Oracle: 27.36 us
- Compile (cd=True, combo=True): 26.21 us
- Ratio: 0.958 (compile is FASTER than oracle)
- Status: AT_FLOOR

## Diagnosis

Inductor matches the oracle for this head layout view pattern on [512, 64, 512] tensor with non-trivial stride requirements. No gap to investigate.
