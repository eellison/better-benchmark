# var_mean_b31c33bdc684

## Classification: BANDWIDTH_BOUND

## Current Result

- Family: `dropout_residual_layernorm`
- Oracle path: `repros/canonical/var_mean_b31c33bdc684/oracle_dropout_residual_layernorm.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: `42.91 us`
- `torch.compile coordinate_descent_tuning=True`: `51.1 us`
- Ratio: 1.191
- Best config: `combo+mk=3`: `53.30 us` (slightly worse), `combo+mk=2`: `55.02 us` (worse)
- Status: `real_gap`

## Diagnosis

The oracle computes the complete M2M100 training dropout-residual LayerNorm scope in one Triton row-reduction kernel for [8192, 1024], including internally generated Inductor seed-index-0 dropout, residual add, fp32 population var_mean over hidden size 1024, eps=1e-5 rsqrt, affine scale/bias, and final contiguous [8192,1024] view.

The oracle's docstring self-classifies as BANDWIDTH_BOUND, stating tuned Inductor already emits a fused persistent norm kernel for the same full scope. However, the measured 1.19x gap suggests Inductor's norm template is not fully optimal for this particular hidden size / dropout combination.

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True) | 51.1 |
| combo+mk=2 | 55.02 (worse) |
| combo+mk=3 | 53.30 (worse) |
| Oracle | 42.91 |

No config helps -- multi-kernel makes it worse.

## Root cause

The persistent norm template's autotuning for hidden_size=1024 with RNG dropout appears suboptimal. The oracle uses a hand-tuned block size and register allocation that achieves better occupancy for this specific shape.

## Kernel count
- Oracle: 1 kernel (fused dropout + residual + LN)
- Inductor: 1 kernel (same fusion, but suboptimal tuning)

## Recommendation

The gap is likely in autotune heuristics for persistent reduction with fused RNG. The oracle achieves better occupancy through manual block size selection. Could be addressed by expanding the autotuning search space in `torch/_inductor/choices.py` for norm templates with RNG.
