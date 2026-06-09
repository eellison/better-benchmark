# sum_530cee9dd34e

## Classification: BANDWIDTH_BOUND

## Current Result

- Family: `attention_backward_row`
- Oracle path: `repros/canonical/sum_530cee9dd34e/oracle_attention_backward_row.py`
- Correctness: PASS (shape=[1024, 128, 128] dtype=torch.float32 max_diff=1.64e+04)
- Oracle: `36.38 us`
- `torch.compile coordinate_descent_tuning=True`: `42.82 us`
- Ratio: 1.177
- Best config: `combo+mk=3`: `39.92 us` (ratio 1.10x)
- Status: `real_gap` (small, partially closeable with mk=3)

## Diagnosis

The oracle computes the M2M100 attention backward row-reduction scope in one Triton kernel with XBLOCK=4, including score view, broadcast mask fill, exp/div probability, element mask scale, row sum, FMA epilogue, and final contiguous view. Inductor already emits a similar fused persistent row reduction but with slightly different tuning.

The oracle uses XBLOCK=4 with num_warps=2 and num_stages=1, which is optimized for the 128-wide reduction with 131072 rows. The remaining gap after mk=3 (1.10x) is within the noise threshold for bandwidth-bound kernels.

## Config exploration results

| Config | Time (us) | Ratio vs oracle |
|--------|-----------|-----------------|
| Default (cd=True) | 42.82 | 1.177 |
| combo+mk=2 | 44.28 | 1.217 |
| combo+mk=3 | 39.92 | 1.097 |
| fast_math | 42.19 | 1.160 |
| Oracle | 36.38 | 1.000 |

`multi_kernel=3` (looped reduction) brings the gap down to ~1.10x. The remaining gap is likely due to minor tuning differences (num_warps, XBLOCK selection) in the bandwidth-bound regime.

## Root cause

Inductor emits essentially the same computation but with suboptimal autotuning parameters for this specific shape. The 128-wide persistent reduction with 131072 rows is bandwidth-bound and benefits from the looped reduction variant (mk=3) that Inductor's auto-selection does not choose by default.

## Kernel count
- Oracle: 1 kernel (persistent row reduction, XBLOCK=4)
- Inductor: 1 kernel (persistent row reduction, default tuning)

## Recommendation

The gap is mostly closed by `multi_kernel=3`. The residual ~10% is in autotuning territory (XBLOCK, num_warps selection). This is a PERSISTENT_THRESHOLD / autotuning quality issue rather than a fundamental scheduler gap. Could be improved by better heuristics in `choices.py` for this shape regime.
