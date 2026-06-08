# var_mean_b551e8ee78b4

## Classification: NO_GAP (BAD_ORACLE)

## Current Result

- Family: `swin_window_layernorm`
- Oracle path: `repros/canonical/var_mean_b551e8ee78b4/oracle_swin_window_layernorm.py`
- Correctness: PASS (shape=[25088, 512] dtype=torch.float32 max_diff=2.86e-06)
- Oracle: `23.33 us`
- `torch.compile coordinate_descent_tuning=True`: `21.92 us`
- Ratio: 0.940
- Status: `bad_oracle` (compile is 6% faster)

## Diagnosis

The oracle is slower than `torch.compile`. Inductor already generates optimal code for this Swin window LayerNorm pattern. No gap exists.

## Kernel count
- Oracle: 1 kernel
- Inductor: equivalent or better

## Recommendation

No action needed. The oracle does not demonstrate a gap.
