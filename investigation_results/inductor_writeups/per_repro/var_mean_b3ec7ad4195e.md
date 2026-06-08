# var_mean_b3ec7ad4195e

## Classification: NO_GAP (AT_FLOOR)

## Current Result

- Family: `add_layernorm_saved_scale`
- Oracle path: `repros/canonical/var_mean_b3ec7ad4195e/oracle_add_layernorm_saved_scale.py`
- Correctness: PASS (shape=[4096, 4096] dtype=torch.float32, shape=[8, 512, 1] dtype=torch.float32)
- Oracle: `38.50 us`
- `torch.compile coordinate_descent_tuning=True`: `37.60 us`
- Ratio: 0.977
- Status: `at_floor` (compile matches or beats oracle)

## Diagnosis

Inductor already matches the oracle performance. The compile output (37.60us) is actually slightly faster than the oracle (38.50us). No gap exists.

## Kernel count
- Oracle: 1 kernel
- Inductor: equivalent

## Recommendation

No action needed. Inductor is at floor for this pattern.
