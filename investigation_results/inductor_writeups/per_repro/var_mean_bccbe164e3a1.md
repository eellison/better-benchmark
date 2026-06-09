# var_mean_bccbe164e3a1

## Classification: NO_GAP (AT_FLOOR)

## Current Result

- Family: `add_layernorm`
- Oracle path: `repros/canonical/var_mean_bccbe164e3a1/oracle_add_layernorm.py`
- Correctness: PASS (shape=[4096, 4096] dtype=torch.float32 max_diff=2.86e-06)
- Oracle: `38.59 us`
- `torch.compile coordinate_descent_tuning=True`: `36.80 us`
- Ratio: 0.954
- Status: `at_floor` (compile matches or beats oracle)

## Diagnosis

Inductor already matches or beats the oracle performance (36.80us vs 38.59us). No gap exists for this add + LayerNorm pattern.

## Kernel count
- Oracle: 1 kernel
- Inductor: equivalent

## Recommendation

No action needed. Inductor is at floor for this pattern.
