# var_mean_b815b1a264a5

## Classification: BAD_ORACLE

## Current Result

- Family: `swin_indexed_window_layernorm`
- Oracle path: `repros/canonical/var_mean_b815b1a264a5/oracle_swin_indexed_window_layernorm.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: `69.44 us`
- `torch.compile coordinate_descent_tuning=True`: `43.9 us`
- Ratio: 0.632 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the Swin indexed window LayerNorm pattern is 37% slower than Inductor's generated code on this hardware. The oracle's fused kernel approach with window indexing and dropout is significantly outperformed by Inductor's multi-kernel strategy for this shape.

## Config exploration results
- No configs needed -- oracle is 1.58x slower than baseline compile.
