# var_mean_5e94b32b5fac

## Classification: AT_FLOOR

## Current Result

- Family: `swin_droppath_shifted_window_layernorm`
- Oracle path: `repros/canonical/var_mean_5e94b32b5fac/oracle_swin_droppath_shifted_window_layernorm.py`
- Correctness: PASS (Inductor RNG replayed)
- Oracle: `61.09 us`
- `torch.compile coordinate_descent_tuning=True`: `61.15 us`
- Ratio: 1.001
- Status: `at_floor`

## Diagnosis

Inductor matches the oracle within noise. This Swin shifted-window LayerNorm variant (different from var_mean_00824117c097) is already at the bandwidth floor. The difference from the 00824117c097 variant may be due to different window sizes or shift amounts.

## Config exploration results
- No improvement needed -- already at floor (ratio = 1.001).
