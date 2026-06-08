# var_mean_bfedec863d1b

## Classification: NO_GAP (AT_FLOOR)

## Current Result

- Family: `layernorm_affine_aliases`
- Oracle path: `repros/canonical/var_mean_bfedec863d1b/oracle_layernorm_affine_aliases.py`
- Correctness: PASS (shape=[32768, 256] dtype=torch.float32 x3 outputs max_diff=2.86e-06)
- Oracle: `15.97 us`
- `torch.compile coordinate_descent_tuning=True`: `16.03 us`
- Ratio: 1.004
- Status: `at_floor` (within noise)

## Diagnosis

Inductor matches the oracle within measurement noise (0.4% difference). No gap exists for this LayerNorm with affine aliases pattern.

## Kernel count
- Oracle: 1 kernel
- Inductor: equivalent

## Recommendation

No action needed. Inductor is at floor for this pattern.
