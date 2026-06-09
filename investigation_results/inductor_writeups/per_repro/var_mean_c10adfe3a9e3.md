# var_mean_c10adfe3a9e3

## Classification: NO_GAP (BAD_ORACLE)

## Current Result

- Family: `swin_window_reverse_droppath_layernorm`
- Oracle path: `repros/canonical/var_mean_c10adfe3a9e3/oracle_swin_window_reverse_droppath_layernorm.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: `40.70 us`
- `torch.compile coordinate_descent_tuning=True`: `32.45 us`
- Ratio: 0.797
- Status: `bad_oracle` (compile is 20% faster)

## Diagnosis

The oracle is substantially slower than `torch.compile`. Inductor generates much better code for this Swin window-reverse drop-path LayerNorm pattern. No gap exists.

## Kernel count
- Oracle: 1 kernel
- Inductor: equivalent or better

## Recommendation

No action needed. The oracle does not demonstrate a gap.
