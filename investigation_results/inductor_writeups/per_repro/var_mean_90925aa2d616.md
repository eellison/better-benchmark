# var_mean_90925aa2d616 - oracle_relu_layernorm

## Status: BAD_ORACLE (compile beats oracle)

- Oracle: 31.39 us
- Compile: 26.30 us
- Ratio: 0.838x

## Classification: NO_GAP

Oracle path: `repros/canonical/var_mean_90925aa2d616/oracle_relu_layernorm.py`

The compiled Inductor output outperforms the oracle by 16%. No Inductor improvement needed.

## Details
- Model: hf MobileBertForMaskedLM inference
- Pattern: Contiguous view, NaN-preserving ReLU, hidden-size-512 population var_mean, eps=1e-12 rsqrt normalization, affine scale/bias, contiguous view
- Shape: [32768, 512] f32 output
- Classification in oracle: SCHEDULER_FUSION (sink ReLU producer into LayerNorm template), but Inductor's generic fused reduction already handles this efficiently
- Correctness: PASS
