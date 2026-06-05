# var_mean_abb7e67d11ad

## Classification: SEEDED_DROPOUT_LAYERNORM_SIDE_FUSION

## Current Result

- Family: `seeded_dropout_layernorm_side`
- Oracle path: `repros/canonical/var_mean_abb7e67d11ad/oracle_seeded_dropout_layernorm_side.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: `34.56 us`
- `torch.compile coordinate_descent_tuning=True`: `45.12 us`
- Ratio: 1.306
- Best config: `combo+mk=3`: `73.16 us` (worse); `combo+mk=2`: `76.03 us` (worse)
- Status: `real_gap`

## Diagnosis

The oracle computes the complete M2M100 training seeded-dropout residual LayerNorm scope in one hidden-size-1024 Triton row kernel: [8192,1024] view, seed-index-2 p=0.1 dropout on the projected activation, residual add, fp32 population var_mean (dim=2, correction=0, keepdim=True), eps=1e-5 affine scale/bias, final [8192,1024] view, and sibling rsqrt/1024 side output.

Inductor lowers this norm-template graph into separate RNG, row-reduction, affine, output-store, and side-output operations that do not fully fuse into a single pass.

## Root Cause

Despite the oracle docstring claiming BANDWIDTH_BOUND, the measured 1.31x gap indicates Inductor is NOT fully fusing the dropout+residual+layernorm+side-output into one kernel pass. The oracle achieves this with an autotuned row kernel (ROW_BLOCK=1/2/4) that processes the entire hidden dimension in registers.

The gap likely comes from:
1. Inductor materializing the dropout result before the LayerNorm reduction
2. Separate store operations for the side output (rsqrt/1024)
3. Potential two-pass approach for var_mean (one for mean, one for variance)

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True) | 45.12 |
| combo+mk=2 | 76.03 |
| combo+mk=3 | 73.16 |
| Oracle | 34.56 |

No config closes the gap. Multi-kernel configs significantly worse (they split the single reduction into looped/persistent variants which hurts for this fused pattern).

## Kernel count
- Oracle: 1 kernel (fused dropout + residual + LN + side output)
- Inductor: 2+ kernels (RNG/dropout, then norm reduction + affine + side)

## Fix path

The scheduler/norm-template needs to recognize that seeded-dropout + residual + var_mean + affine + side-output can all be computed in a single row-reduction kernel with inline RNG. The RNG producer should be inlined into the reduction kernel rather than materialized as a separate pointwise op.

File: `/tmp/pytorch-work/torch/_inductor/scheduler.py` (fusion of RNG producer into reduction consumer), `/tmp/pytorch-work/torch/_inductor/ir.py` (realize_hint preventing RNG inline).

## Generalizability

This pattern (dropout + residual + LayerNorm + side output) appears in M2M100, MegatronBERT, and similar transformer models. Multiple var_mean repros in the corpus share this root cause.
