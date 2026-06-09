# var_mean_b3c82aa686ce

## Current Result

- Family: `patch_pos_layernorm`
- Classification: `NEW_PATTERN`
- Oracle path: `repros/canonical/var_mean_b3c82aa686ce/oracle_patch_pos_layernorm.py`
- Correctness: PASS
- Oracle: `67.3 us`
- `torch.compile coordinate_descent_tuning=True`: `71.26 us`
- `combo_kernels + multi_kernel=2`: `71.3 us` (no change)
- `combo_kernels + multi_kernel=3`: `55.0 us` (1.296x improvement, closes gap)
- `coordinate_descent_tuning + use_fast_math`: `71.3 us` (no change)
- Ratio: 1.059x (oracle vs default compile)
- Status: `closed_by_config`

## Diagnosis

The oracle computes the complete ViT patch-token positional-add LayerNorm scope in one Triton row-block kernel, loading the reshape/permute convolution patches, folding the broadcast [1,256,768] positional embedding add, correction=0 var_mean, eps=1e-6 rsqrt, affine epilogue, clone-contiguous layout, and final [32768,768] view. Inductor currently lowers the decomposed view/permute/add/var_mean graph through its generic reduction-normalization path.

## Config exploration results

| Config | Time (us) | vs cd |
|--------|-----------|-------|
| cd (baseline) | 71.3 | 1.000x |
| combo + mk=2 | 71.3 | 1.000x |
| combo + mk=3 | 55.0 | 1.296x faster |
| cd + use_fast_math | 71.3 | 1.000x |
| Oracle | 67.3 | 1.059x faster |

**combo_kernels + multi_kernel=3 fully closes the gap and exceeds the oracle** at 55.0us vs 67.3us oracle (0.818x). The multi_kernel=3 persistent reduction path is highly effective for this hidden_size=768 LayerNorm pattern.

## Root cause

The 5.9% gap at default config comes from Inductor's generic norm-template path not optimally scheduling the permuted NCHW convolution gather + positional broadcast add as a row-blocked pattern. However, the `multi_kernel=3` config (which enables persistent reduction with autotuning) already produces code that significantly outperforms even the hand-written oracle.

## Classification: NEW_PATTERN

The gap is marginal (5.9%) and is fully closed by `multi_kernel=3`. The oracle's `NEW_PATTERN` classification is still valid as a code-quality improvement (recognizing ViT patch-position LayerNorm as a template), but there is no performance gap requiring a fix when the right config is used.

## Recommendation

Enable `triton.multi_kernel=3` for this class of LayerNorm kernels with hidden_size=768 and permuted input gathers. No new pattern template is needed for performance.

## Details
- Model: timm_timm_vit_base_patch16_siglip_256_infer_infer_000
- Pattern: view -> permute -> add(pos_embed) -> var_mean -> sub -> rsqrt -> mul(weight) -> add(bias) -> clone -> view
- Shape: [128,768,16,16] conv -> permute -> [128,256,768] + [1,256,768] pos -> LN -> [32768,768]
- Hidden size: 768, correction=0, eps=1e-6
