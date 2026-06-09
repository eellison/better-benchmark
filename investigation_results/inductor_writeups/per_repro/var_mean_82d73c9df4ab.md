# var_mean_82d73c9df4ab

## Classification: AT_FLOOR

## Current Result

- Family: `swin_window_layernorm`
- Oracle path: `repros/canonical/var_mean_82d73c9df4ab/oracle_swin_window_layernorm.py`
- Correctness: PASS
- Oracle: `14.05 us`
- `torch.compile coordinate_descent_tuning=True`: `13.86 us`
- Ratio: 0.986 (effectively at parity)
- Status: `at_floor`

## Diagnosis

Inductor is within 1.4% of the oracle for this Swin 7x7 window-partition LayerNorm pattern with hidden size 1024. The oracle fuses the full hidden-size-1024 population var_mean LayerNorm affine and singleton Swin window-partition reshape/permute/flatten aliases into one contiguous row kernel. Inductor already achieves equivalent performance through its generic normalization path. No actionable improvement needed.

## Config exploration results
- Baseline is already at parity with oracle (0.986x, oracle slightly slower).
