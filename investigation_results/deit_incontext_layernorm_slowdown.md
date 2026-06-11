# deit_tiny: in-model layernorm runs up to 4x slower than identical standalone shape

**Status:** OPEN (filed 2026-06-11, not blocking wave 1)
**Found by:** `scripts/model_attribution.py` 10-model coherence validation —
the only model whose corrected parts/e2e ratio is materially below 1.0.

## The numbers (B200, bf16 infer, batch from /tmp/wave1_batch5)

- e2e (CUDAGraph, CD on): **2496us**
- Σ(standalone × occurrences), launch-floor corrected: **1839us** → ratio **0.74**
- All other 9 batch models: 0.92–1.08. Zero unmatched occurrences, so this
  is not a coverage gap — the model genuinely runs ~650us (26%) slower than
  its own parts measured standalone.

## Where it lives

In-model profile: the layernorm reduction kernels
(`triton_red_fused_add_convert_element_type_mul_rsqrt_sub_var_mean_view_*`).
Twelve near-identical instances (same pattern hash, same shape `[512, 197,
192]`-class, same dtype) range **15us → 67us, monotonically slower with
layer depth**:

```
layer  1:  15us     layer  7:  49us
layer  3:  17-18us  layer  9:  55-60us
layer  5:  20-35us  layer 11:  64-67us
```

Standalone, the identical (pattern, shape, stride, dtype) point benches at
the fast end (~15-17us). The gelu pointwise kernel (12x 293us total) and
extern gemms do NOT show the depth trend — it's specific to the var_mean
reduction.

## Hypotheses (untested)

1. **Input locality drift:** in-model the layernorm reads the residual
   stream written by the preceding gemm; allocation pattern may push
   later layers' residuals into worse-locality addresses. Standalone
   fabricated inputs can't reproduce this by construction.
2. **L2 contention with neighboring kernels** in the same graph replay
   (standalone graphs contain just the one kernel).
3. Same-kernel-different-launch-context effects under CUDAGraph (stream
   scheduling within the captured graph).

Note: visformer (also attention-based, conv-hybrid) does NOT show the
effect (ratio 1.02), so it is not "all transformers" — but check the hf
suite in wave 1 for recurrence on pure-transformer models.

## Repro

```
CUDA_VISIBLE_DEVICES=0 INDUCTOR_GPU_BENCH_LOCK=1 \
python scripts/model_attribution.py --corpus-root /tmp/wave1_batch5/repros \
  --suite timm --mode infer --models deit_tiny_patch16_224.fb_in1k
```
then profile the e2e graph and compare per-launch times of the
`var_mean` kernels against the standalone point at the same shape hash.
