# var_mean_a1464365da4f

## Status: BAD_ORACLE (compile already faster)

- Oracle: 13.02 us
- Compile: 11.55 us
- Ratio: 0.887x (oracle is 1.13x slower than compile)

## Classification: NO_GAP

Oracle path: `repros/canonical/var_mean_a1464365da4f/oracle_channel_norm_scale.py`

The compiled Inductor output already outperforms this oracle. No Inductor improvement needed. The oracle's shape-specialized kernel for per-channel normalization does not outperform Inductor's generic schedule at this particular shape.

## Previous diagnosis (superseded by measurement)

The oracle was designed to compute NFNet per-channel population var_mean, rsqrt side output, scalar-gain normalized activation, and keepdim mean side output in one shape-specialized Triton program. However, Inductor's default schedule already achieves better performance.

## Details
- Model: timm_dm_nfnet (training)
- Pattern: per-channel var_mean + rsqrt + scale-normalize
- Shape: [3072, 1536, 1, 1] f32, producing [3072] rsqrt, [3072,1536,1,1] normalized, [1,3072,1] mean
- Correctness: PASS
