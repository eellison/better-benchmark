# var_mean_d549c16798a1

## Classification: SCHEDULER_FUSION

## Current Result

- Family: `swin_droppath_layernorm`
- Oracle path: `repros/canonical/var_mean_d549c16798a1/oracle_swin_droppath_layernorm.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: `30.50 us`
- `torch.compile coordinate_descent_tuning=True`: `38.30 us`
- Ratio: 1.256
- Best config: `fast_math=True`: `37.89 us` (ratio 1.208, no meaningful improvement)
- Status: `real_gap`

## Diagnosis

The oracle computes the complete Swin window-reverse drop-path residual LayerNorm scope in one Triton row-reduction kernel: static 7x7 window layout clone (via index arithmetic), seed-index-38 drop-path mask using Inductor RNG, residual add, hidden-size-512 population var_mean, rsqrt epsilon epilogue, affine scale/bias, final [25088,512] view, and live rsqrt/512 side output. It handles 25088 rows (batch=128, 14x14 spatial, 4 windows of 7x7).

Inductor currently materializes the `prims.inductor_random` producer as a separate pointwise kernel before fusing the window-reverse clone, drop-path residual, and normalization into its reduction template. The materialization boundary at the RNG producer creates a kernel launch overhead and forces an extra global memory write/read for the dropout mask.

## Config exploration results

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Default (cd=True) | 38.30 | 1.256 |
| multi_kernel=2 | 37.57 | 1.203 |
| multi_kernel=3 | 37.79 | 1.198 |
| fast_math=True | 37.89 | 1.208 |
| combo (mk=2 + fast_math) | 37.66 | 1.201 |
| Oracle | 30.50 | 1.000 |

No config closes the 1.25x gap. Multi-kernel=2/3 provides marginal improvement (~5%) but the 20%+ gap remains.

## Root cause

The normalization scheduler treats Inductor RNG (`prims.inductor_random`) as a materialized producer boundary for normalization reductions. It cannot inline the seed/offset expression into the row-reduction schedule. This forces:
1. A separate kernel launch for the RNG producer (seed lookup + rand + lt + convert + div)
2. A global memory round-trip for the [128,1,1,1] drop-path mask
3. The window-reverse index arithmetic to be separated from the normalization body

The oracle avoids all three by inlining `tl.rand(seed, batch_id)` directly in the row-reduction kernel and computing window-to-spatial index mapping arithmetically.

## Kernel count

- Oracle: 1 kernel (fused window-reverse + drop-path RNG + residual + LN + side output)
- Inductor: 2+ kernels (RNG producer pointwise + norm reduction with window-reverse)

## Recommendation

Requires teaching the normalization scheduler to fuse fixed-shape seeded RNG producers with batch-broadcast uses into the LayerNorm reduction template. The pattern is: `inductor_lookup_seed` + `inductor_random([B,1,1,1])` + `lt.Scalar` + `convert_element_type` + `div.Tensor` produces a per-batch stochastic scale that broadcasts into the row reduction. The scheduler should recognize this as an inlineable scalar producer rather than a materialization boundary.

Same family as var_mean_00824117c097 (shifted-window variant) and var_mean_dd3ccbc569e6 (same pattern, different shape achieving floor).

## Relevant files

- `repros/canonical/var_mean_d549c16798a1/repro.py` (Swin train drop-path + window-reverse + LN)
- `repros/canonical/var_mean_d549c16798a1/oracle_swin_droppath_layernorm.py` (oracle)
