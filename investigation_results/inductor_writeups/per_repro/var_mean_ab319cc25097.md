# var_mean_ab319cc25097

## Classification: EMBEDDING_LAYERNORM_KERNEL_SPLIT

## Current Result

- Family: `embedding_layernorm`
- Oracle path: `repros/canonical/var_mean_ab319cc25097/oracle_embedding_layernorm.py`
- Correctness: PASS
- Oracle: `16.16 us`
- `torch.compile coordinate_descent_tuning=True`: `17.15 us`
- Ratio: 1.061
- Best config: `default (cd=True)`: `17.15 us`
- Status: `real_gap` (marginal)

## Diagnosis

The oracle computes embedding lookup + LayerNorm (var_mean + affine) fused into one kernel on shape [2048, 2560], producing 3 outputs. Inductor splits this into 2 kernels:
1. `triton_poi_fused_embedding_iota_0` -- embedding lookup pointwise
2. `triton_per_fused_add_embedding_iota_mul_rsqrt_sub_var_mean_1` -- add + LayerNorm reduction

The oracle fuses the embedding lookup into the LayerNorm reduction kernel directly, avoiding the intermediate materialization.

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True) | 17.15 |
| combo+mk=2 | 29.83 |
| combo+mk=3 | 27.83 |
| Oracle | 16.16 |

No config closes the gap. Multi-kernel configs make it significantly worse (higher overhead from different reduction strategy on this shape).

## Root cause

The 6.1% gap comes from the embedding lookup being scheduled as a separate kernel rather than being inlined into the LayerNorm reduction. The embedding lookup is a simple gather operation that could be fused as a prologue to the reduction, but Inductor materializes it because the embedding output is used by multiple consumers (the 3 output paths share it).

## Kernel count
- Oracle: 1 kernel (fused embedding + LN + 3 outputs)
- Inductor: 2 kernels (embedding pointwise, then LN reduction with 3 outputs)

## Recommendation

The gap is marginal (6.1%). The fix would require the scheduler to inline the embedding gather into the reduction kernel when the embedding output has no other external consumers besides the reduction's input. This is a producer-inline optimization for gather/embedding ops.

File references: `torch/_inductor/scheduler.py` (fusion of gather producers into reduction consumers).
