# var_mean_ccc838fc023d

## Classification: SCHEDULER_FUSION

## Current Result

- Family: `swin_window_layernorm`
- Oracle path: `repros/canonical/var_mean_ccc838fc023d/oracle_swin_window_layernorm.py`
- Correctness: PASS (max_diff=2.86e-06)
- Oracle: `29.18 us`
- `torch.compile coordinate_descent_tuning=True`: `29.41 us`
- Ratio: 1.008
- Status: `at_floor`

## Diagnosis

The oracle computes the residual add, hidden-size-512 population var_mean LayerNorm affine, and fixed Swin 7x7 window-partition clone in one output-contiguous Triton row kernel. Inductor currently fuses the same scope but traverses source rows and emits strided window-layout stores from the normalization epilogue.

However, the measured performance gap is negligible (0.8%). Inductor's codegen for this inference-only Swin window-reverse LayerNorm pattern at 14x14 spatial resolution is effectively at parity with the hand-written oracle.

## Root cause

No actionable gap. The oracle describes a theoretical scheduler fusion improvement (output-contiguous vs source-contiguous epilogue traversal), but Inductor's current codegen already achieves oracle-level performance for this shape. The 0.8% difference is within measurement noise.

## Kernel count

- Oracle: 1 kernel (fused residual + LN + window-partition)
- Inductor: 1 kernel (same fusion, different traversal order)

## Recommendation

No Inductor change needed. Record as at bandwidth floor. The SCHEDULER_FUSION classification in the oracle docstring identifies a real codegen difference (traversal order), but it does not manifest as a performance gap for this specific shape (25088 rows x 512 hidden, Swin inference at 14x14).

## Relevant files

- `repros/canonical/var_mean_ccc838fc023d/repro.py` (Swin infer window-partition LN)
- `repros/canonical/var_mean_ccc838fc023d/oracle_swin_window_layernorm.py` (oracle)
