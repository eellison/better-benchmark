# var_mean_dd3ccbc569e6

## Classification: SCHEDULER_FUSION

## Current Result

- Family: `swin_droppath_layernorm`
- Oracle path: `repros/canonical/var_mean_dd3ccbc569e6/oracle_swin_droppath_layernorm.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: `19.30 us`
- `torch.compile coordinate_descent_tuning=True`: `19.42 us`
- Ratio: 1.007
- Status: `at_floor`

## Diagnosis

The oracle fuses the full Swin window-reverse drop-path residual LayerNorm scope (seed-index-44, hidden-size-1024, population var_mean, affine epilogue, final flatten, rsqrt/1024 side output) into one shape-specialized row reduction kernel. Inductor currently materializes the `prims.inductor_random` in a separate pointwise kernel before its fused var_mean kernel.

However, despite the theoretical scheduler fusion issue (same as var_mean_d549c16798a1), the measured performance gap is negligible (0.7%). At this particular shape, Inductor's multi-kernel approach already achieves oracle-level performance.

## Root cause

No actionable gap at this shape. The SCHEDULER_FUSION issue (RNG materialization boundary) identified in the oracle docstring is real in principle but does not manifest as a measurable performance difference here. This is likely because the kernel is small enough that the extra kernel launch overhead is hidden by GPU occupancy.

## Kernel count

- Oracle: 1 kernel (fused window-reverse + drop-path RNG + residual + LN)
- Inductor: 2+ kernels (RNG producer + norm reduction)

## Recommendation

No Inductor change needed for this specific shape. Record as at floor. The same underlying pattern (Swin drop-path + window-reverse + LN) shows a real gap at the larger shape in var_mean_d549c16798a1 (25088 rows, ratio 1.256) but not here. Fixing the RNG producer fusion for the sibling repro would also benefit this one, though it is already at parity.

## Relevant files

- `repros/canonical/var_mean_dd3ccbc569e6/repro.py` (Swin train drop-path + window-reverse + LN)
- `repros/canonical/var_mean_dd3ccbc569e6/oracle_swin_droppath_layernorm.py` (oracle)
