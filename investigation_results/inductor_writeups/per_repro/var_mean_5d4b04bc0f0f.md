# var_mean_5d4b04bc0f0f

## Classification: AT_FLOOR

## Current Result

- Family: `selected_class_token_layernorm`
- Oracle path: `repros/canonical/var_mean_5d4b04bc0f0f/oracle_selected_class_token_layernorm.py`
- Correctness: PASS
- Oracle: `6.08 us`
- `torch.compile coordinate_descent_tuning=True`: `5.98 us`
- Ratio: 0.984 (oracle slightly slower)
- Status: `at_floor`

## Diagnosis

Inductor matches or beats the oracle. The selected class-token LayerNorm for [128, 192] is at the launch overhead floor. Same family as var_mean_4018778d8833.

## Config exploration results
- No improvement needed -- already at floor.
