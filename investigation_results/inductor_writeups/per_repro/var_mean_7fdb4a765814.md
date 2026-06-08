# var_mean_7fdb4a765814 - Strided Add LayerNorm (ViT SigLIP)

## Classification: AT_FLOOR

## Benchmark Results
- Oracle: 80.74 us
- Compile (baseline): 82.85 us
- Ratio: 1.026x (within noise, no meaningful gap)
- Status: AT_FLOOR

## Oracle
- Path: `repros/canonical/var_mean_7fdb4a765814/oracle_strided_add_layernorm.py`
- Correctness: PASS (max_diff=2.86e-06)
- Model: timm_vit_base_patch16_siglip_256 (inference)
- Shape: [32768, 768] (batch=128, seq=256, hidden=768)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The oracle fuses
a strided residual add, population var_mean over hidden size 768 with eps=1e-6, rsqrt
normalization, fp32 affine scale/bias, contiguous clone materialization, and final
[32768, 768] view into one multi-row Triton kernel. However, Inductor's generic
reduction schedule achieves near-identical performance because the kernel is
memory-bandwidth-bound on the large [32768, 768] tensor and the scheduling
overhead of the non-contiguous residual input is negligible at this row count.

## Config Exploration
- No further config exploration needed (baseline already at oracle level).
