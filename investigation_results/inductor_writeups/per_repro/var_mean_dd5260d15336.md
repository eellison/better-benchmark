# var_mean_dd5260d15336

## Classification: BAD_ORACLE

## Current Result

- Family: `layernorm_affine`
- Oracle path: `repros/canonical/var_mean_dd5260d15336/oracle_layernorm_affine.py`
- Correctness: PASS (max_diff=1.91e-06)
- Oracle: `19.17 us`
- `torch.compile coordinate_descent_tuning=True`: `15.74 us`
- Ratio: 0.821 (BAD_ORACLE)

## Diagnosis

The oracle attempts to compute the complete fp32 hidden-size-256 population var_mean, eps=1e-12 rsqrt normalization, affine scale/bias, and final contiguous [32768, 256] view in one static row-blocked Triton LayerNorm kernel. However, Inductor's generic persistent-reduction codegen is significantly faster (18% faster) than the hand-written oracle for this workload.

The oracle is strictly slower than torch.compile output (0.821x) -- there is no performance gap to close. Inductor handles this simple LayerNorm-affine pattern (Reformer model, 32768 rows x 256 hidden) very efficiently.

## Root cause

The oracle's static row-blocked approach is suboptimal compared to Inductor's auto-tuned persistent-reduction kernel for this relatively narrow hidden dimension (256). At K=256, the entire row fits easily in registers and Inductor's XBLOCK/RBLOCK tuning likely finds a better occupancy/throughput tradeoff than the oracle's fixed configuration. The NEW_PATTERN classification in the oracle docstring hypothesized that a specialized template would be faster, but Inductor's generic codegen already outperforms it.

## Kernel count

- Oracle: 1 kernel
- Inductor: 1 kernel (faster)

## Recommendation

Record as BAD_ORACLE. The oracle needs to be rewritten to be competitive for this shape, or this repro should be deprioritized. No Inductor change needed -- Inductor already wins by a significant margin (18%).

## Relevant files

- `repros/canonical/var_mean_dd5260d15336/repro.py` (Reformer LayerNorm affine, K=256)
- `repros/canonical/var_mean_dd5260d15336/oracle_layernorm_affine.py` (oracle)
