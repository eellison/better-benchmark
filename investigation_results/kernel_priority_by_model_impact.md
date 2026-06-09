# Kernel Fix Priority Ranking by Model-Level Impact

Date: 2026-06-09

## Executive Summary

This analysis ranks gap CLASSIFICATIONS by their total weighted impact across
all models in the benchmark corpus. The metric answers: **"If we fix classification X,
what is the total fusible-portion speedup summed across all affected models?"**

### Methodology

For each repro with a measured gap:
1. Look up its classification (from `inductor_optimization_per_repro_queue.csv`)
2. Find which models contain it (via `manifest.json` pattern hash mappings)
3. Compute `kernel_gap / model_total_fusible_time` = fraction of model's fusible time saved
4. Sum these fractions across all affected models = total weighted impact

**Gap definition**: `best_compile_us - (memcopy_sol_us + 3us * n_kernels)` = excess time
over the launch-adjusted speed-of-light. This is a CONSERVATIVE estimate that assumes
the speed-of-light (memory-copy bandwidth) is achievable for each kernel.

### Key Caveat: Fusible vs End-to-End

These numbers are for the **fusible portion only** (pointwise, reductions, norms).
Non-fusible ops (matmul, conv, SDPA, embedding) often dominate total runtime.
For models where fusible ops are <5% of total time, even a 50% fusible speedup
translates to <2.5% end-to-end improvement.

---

## Priority Ranking

| Rank | Classification | Repros | Models | Total Gap (us) | Weighted Impact | Avg/Model |
|------|---------------|--------|--------|---------------|-----------------|-----------|
| 1 | **multi_output_reduction_templates** | 245 | 82 | 61,514 | **1785.8%** | 21.8% |
| 2 | **online_softmax_cross_entropy** | 90 | 64 | 35,849 | **1331.2%** | 20.8% |
| 3 | **layout_indexing_stencil_fusion** | 118 | 70 | 11,798 | **1309.6%** | 18.7% |
| 4 | **structured_pool_upsample_backward_reduce** | 112 | 54 | 55,555 | **945.6%** | 17.5% |
| 5 | **norm_template_canonicalization** | 278 | 132 | 11,548 | **612.4%** | 4.6% |
| 6 | **irregular_gather_reduce** | 1 | 1 | 2,264 | **29.4%** | 29.4% |
| 7 | **pointwise** | 2 | 4 | 19 | **3.4%** | 0.9% |
| 8 | **var_mean** | 2 | 1 | 18 | **1.5%** | 1.5% |
| 9 | **sum** | 2 | 9 | 42 | **1.1%** | 0.1% |
| 10 | **any_amax_sum** | 1 | 1 | 3 | **0.3%** | 0.3% |
| 11 | **sum_sum** | 2 | 3 | 8 | **0.3%** | 0.1% |
| 12 | **mean** | 1 | 1 | 2 | **0.0%** | 0.0% |

**Interpretation**: "Weighted Impact" = sum of (gap/model_fusible_time) across all
affected models, expressed as percentage points. Higher = more total speedup delivered.

---

## Top 5 Classifications: Detailed Breakdown

### 1. multi_output_reduction_templates

- **Kernel repros with this gap**: 245
- **Models affected**: 82
- **Total gap across all instances**: 61,514 us
- **Total weighted model impact**: 1785.8 percentage points
- **Average impact per affected model**: 21.8% fusible speedup

**What it is**: Multiple reductions over the same input computed separately instead of in one pass. Example: sum(x, dim=0) and sum(x**2, dim=0) done as two kernels reading x twice, when a single Welford-style pass would halve memory traffic. Affects BN-backward, norm statistics, and any multi-output reduction pattern.

**Top 10 models that benefit most:**

| Model | Fusible Speedup | Gap (us) | Fusible % of Total | End-to-End Impact |
|-------|----------------|----------|-------------------|-------------------|
| timm_convnextv2_nano.fcmae_ft_in22k_in1k_train | 69.1% | 2050 | N/A | N/A |
| torchbench_resnet18_train | 60.0% | 2434 | 11.1% | ~6.7% |
| timm_adv_inception_v3_train | 56.4% | 6257 | N/A | N/A |
| timm_inception_v3_train | 56.4% | 6257 | N/A | N/A |
| torchbench_mobilenet_v2_train | 53.2% | 471 | 0.4% | ~0.2% |
| torchbench_hf_T5_base_train | 52.9% | 5753 | 9.5% | ~5.0% |
| torchbench_mobilenet_v3_large_train | 49.9% | 1489 | 0.8% | ~0.4% |
| timm_mobilevit_s_train | 48.0% | 1191 | N/A | N/A |
| torchbench_mnasnet1_0_train | 47.9% | 506 | 0.3% | ~0.1% |
| torchbench_phlippe_densenet_train | 47.5% | 432 | 30.5% | ~14.5% |

**Sample repro IDs** (first 10 of 245):

- `sum_011e69da166d`
- `sum_031c8d6c51f7`
- `sum_0becf9609ad7`
- `sum_0c674ef4b13c`
- `sum_21c4018e2b7b`
- `sum_27f8a4b9ab09`
- `sum_2bcc7099936f`
- `sum_2e0e9617102b`
- `sum_31bf563cdf96`
- `sum_33b494a50932`

---

### 2. online_softmax_cross_entropy

- **Kernel repros with this gap**: 90
- **Models affected**: 64
- **Total gap across all instances**: 35,849 us
- **Total weighted model impact**: 1331.2 percentage points
- **Average impact per affected model**: 20.8% fusible speedup

**What it is**: Two-pass softmax (compute max+sum, then normalize) and cross-entropy that materializes the full softmax output. An online/streaming algorithm (FlashAttention-style) can compute these in a single pass without materializing the large intermediate. Affects attention layers and vocabulary-sized CE losses.

**Top 10 models that benefit most:**

| Model | Fusible Speedup | Gap (us) | Fusible % of Total | End-to-End Impact |
|-------|----------------|----------|-------------------|-------------------|
| torchbench_hf_T5_train | 76.6% | 6855 | 24.1% | ~18.4% |
| genai_CrossEntropyForward | 71.8% | 1597 | N/A | N/A |
| genai_SoftmaxForward | 64.3% | 2248 | N/A | N/A |
| torchbench_hf_T5_base_infer | 59.8% | 341 | 15.6% | ~9.3% |
| torchbench_hf_T5_infer | 59.5% | 324 | 68.7% | ~40.9% |
| hf_AllenaiLongformerBase_infer | 53.6% | 835 | 7.4% | ~4.0% |
| torchbench_hf_Longformer_infer | 52.9% | 267 | 45.1% | ~23.9% |
| hf_DistillGPT2_infer | 49.5% | 1225 | 5.1% | ~2.5% |
| genai_CrossEntropyBackward | 42.2% | 1619 | N/A | N/A |
| genai_SoftmaxBackward | 42.1% | 2628 | N/A | N/A |

**Sample repro IDs** (first 10 of 90):

- `amax_sum_0561785713ab`
- `amax_sum_0f87f6568d0f`
- `amax_sum_11133e3f9217`
- `amax_sum_17ab35828f89`
- `amax_sum_1a6624550d06`
- `amax_sum_230cc68d5501`
- `amax_sum_35490be2986b`
- `amax_sum_3ed297ef02cd`
- `amax_sum_4b162354068b`
- `amax_sum_4bd27b112605`

---

### 3. layout_indexing_stencil_fusion

- **Kernel repros with this gap**: 118
- **Models affected**: 70
- **Total gap across all instances**: 11,798 us
- **Total weighted model impact**: 1309.6 percentage points
- **Average impact per affected model**: 18.7% fusible speedup

**What it is**: Pointwise producers cannot be inlined into stencil/pool consumers because the consumer reads multiple spatially-shifted positions. BN+ReLU writes a large intermediate that maxpool then re-reads 9 times (3x3 window). Fixing requires either: (a) sinking the producer into the stencil loop, or (b) tiling so the producer output stays in shared memory.

**Top 10 models that benefit most:**

| Model | Fusible Speedup | Gap (us) | Fusible % of Total | End-to-End Impact |
|-------|----------------|----------|-------------------|-------------------|
| torchbench_pyhpc_equation_of_state_infer | 84.1% | 71 | N/A | N/A |
| torchbench_squeezenet1_1_infer | 75.2% | 929 | 45.8% | ~34.4% |
| torchbench_alexnet_infer | 73.8% | 201 | 27.8% | ~20.5% |
| timm_adv_inception_v3_infer | 72.8% | 2321 | N/A | N/A |
| timm_inception_v3_infer | 72.8% | 2321 | N/A | N/A |
| torchbench_pyhpc_isoneutral_mixing_infer | 70.2% | 664 | N/A | N/A |
| torchbench_pytorch_unet_infer | 61.9% | 1008 | 4.4% | ~2.7% |
| torchbench_shufflenet_v2_x1_0_infer | 60.1% | 193 | 18.2% | ~11.0% |
| torchbench_densenet121_infer | 55.4% | 167 | 53.4% | ~29.6% |
| torchbench_resnet152_infer | 51.9% | 71 | 4.3% | ~2.2% |

**Sample repro IDs** (first 10 of 118):

- `pointwise_002b387bc089`
- `pointwise_03cf69441b22`
- `pointwise_083e5f8c7cdc`
- `pointwise_08701284b477`
- `pointwise_0a306c604828`
- `pointwise_0dbfa02ebb68`
- `pointwise_106406131986`
- `pointwise_1383c536f620`
- `pointwise_13f26d420251`
- `pointwise_14c6968f5870`

---

### 4. structured_pool_upsample_backward_reduce

- **Kernel repros with this gap**: 112
- **Models affected**: 54
- **Total gap across all instances**: 55,555 us
- **Total weighted model impact**: 945.6 percentage points
- **Average impact per affected model**: 17.5% fusible speedup

**What it is**: Pool/upsample backward ops use scatter/index_put to build a dense gradient buffer, then reduce it. The scatter materializes a huge tensor that could be avoided by computing the reduction directly from the sparse gradient sources. Affects CNN training (maxpool backward, upsample backward).

**Top 10 models that benefit most:**

| Model | Fusible Speedup | Gap (us) | Fusible % of Total | End-to-End Impact |
|-------|----------------|----------|-------------------|-------------------|
| torchbench_pytorch_unet_train | 78.4% | 23848 | 44.5% | ~34.9% |
| torchbench_vgg16_train | 77.8% | 1835 | 5.7% | ~4.4% |
| torchbench_alexnet_train | 57.5% | 1050 | 10.4% | ~6.0% |
| torchbench_squeezenet1_1_train | 44.5% | 2129 | 15.5% | ~6.9% |
| hf_GPTJForQuestionAnswering_train | 41.9% | 228 | 0.1% | ~0.0% |
| hf_GPTJForCausalLM_train | 38.2% | 228 | 0.5% | ~0.2% |
| hf_AllenaiLongformerBase_train | 37.2% | 2875 | 13.3% | ~4.9% |
| hf_LayoutLMForMaskedLM_train | 35.1% | 2468 | 0.3% | ~0.1% |
| torchbench_hf_Longformer_train | 32.4% | 962 | 7.6% | ~2.5% |
| timm_tf_efficientnet_b0_train | 29.6% | 802 | N/A | N/A |

**Sample repro IDs** (first 10 of 112):

- `sum_046be72c7ad9`
- `sum_0d445802ccc4`
- `sum_1457ecad8f5c`
- `sum_14fe7b321763`
- `sum_18262b26934c`
- `sum_27da2a9850b6`
- `sum_28b0b21169d4`
- `sum_2ccb3947c2c9`
- `sum_34d3dca078b3`
- `sum_3ee4028cab37`

---

### 5. norm_template_canonicalization

- **Kernel repros with this gap**: 278
- **Models affected**: 132
- **Total gap across all instances**: 11,548 us
- **Total weighted model impact**: 612.4 percentage points
- **Average impact per affected model**: 4.6% fusible speedup

**What it is**: BN/LN/RMSNorm chains where the reduction (var+mean) is fast but the surrounding pointwise and stencil operations dominate. Per-channel statistics (rsqrt) may be recomputed N*H*W times due to flat tiling. Many of these are bandwidth-bound and the gap is small per-kernel, but the sheer count (278 repros) makes the aggregate significant.

**Top 10 models that benefit most:**

| Model | Fusible Speedup | Gap (us) | Fusible % of Total | End-to-End Impact |
|-------|----------------|----------|-------------------|-------------------|
| timm_deit_base_distilled_patch16_224_infer | 26.1% | 95 | N/A | N/A |
| torchbench_mnasnet1_0_infer | 24.7% | 16 | 2.2% | ~0.5% |
| timm_convnextv2_nano.fcmae_ft_in22k_in1k_infer | 24.0% | 40 | N/A | N/A |
| timm_ghostnet_100_infer | 23.8% | 83 | N/A | N/A |
| timm_vit_base_patch16_siglip_256_infer | 20.5% | 80 | N/A | N/A |
| torchbench_hf_Whisper_infer | 19.4% | 29 | 51.0% | ~9.9% |
| torchbench_hf_distil_whisper_infer | 19.4% | 29 | 0.0% | ~0.0% |
| timm_vit_base_patch14_dinov2.lvd142m_infer | 19.1% | 229 | N/A | N/A |
| torchbench_mobilenet_v3_large_infer | 17.5% | 30 | 6.8% | ~1.2% |
| genai_LayerNormBackward | 16.9% | 119 | N/A | N/A |

**Sample repro IDs** (first 10 of 278):

- `mean_00c6f66d2b16`
- `mean_03851ba8cf43`
- `mean_148ad2fc17ca`
- `mean_1b78c675fe8a`
- `mean_27e29b31cd8d`
- `mean_289ef979584b`
- `mean_2cec1142a8d5`
- `mean_345e85b88ce8`
- `mean_42b19743bdd2`
- `mean_432300e645a8`

---

## Aggregate Projection: Fix Impact Summary

If each classification were fully resolved, the projected aggregate fusible speedup
(summed across all affected models):

| Classification | Total Fusible Speedup (sum across models) | # Models Improved |
|---------------|------------------------------------------|-------------------|
| multi_output_reduction_templates | +1785.8% total (21.8% avg/model) | 82 |
| online_softmax_cross_entropy | +1331.2% total (20.8% avg/model) | 64 |
| layout_indexing_stencil_fusion | +1309.6% total (18.7% avg/model) | 70 |
| structured_pool_upsample_backward_reduce | +945.6% total (17.5% avg/model) | 54 |
| norm_template_canonicalization | +612.4% total (4.6% avg/model) | 132 |
| irregular_gather_reduce | +29.4% total (29.4% avg/model) | 1 |
| pointwise | +3.4% total (0.9% avg/model) | 4 |

**Combined**: If the top 5 classifications were all fixed, the total aggregate
fusible speedup would be ~5985 percentage points across 50+ models.

---

## Interpretation Guide

- **Total Gap (us)**: Raw microseconds of excess compute across all instances of this pattern.
  Note: a pattern appearing in 5 models counts its gap 5 times.
- **Weighted Impact**: Accounts for how much of each model's fusible time is consumed
  by this pattern. A 100us gap in a model with 200us total fusible time contributes
  50%, while 100us in a model with 10,000us total contributes 1%.
- **Fusible % of Total**: From non-fusible ops accounting. Many transformer models
  have fusible < 1% of total time (matmul dominates). CNN training models often have
  fusible = 5-20% of total time, making fusible speedups more impactful end-to-end.
- **End-to-End Impact**: `fusible_speedup * fusible_fraction_of_total`. This is the
  actual wall-clock improvement users would see.

## Data Sources

- Classification: `investigation_results/inductor_optimization_per_repro_queue.csv`
- Kernel timing: `investigation_results/baseline_results.csv`
- Model-repro mapping: `repros/models/**/manifest.json` (440 manifests)
- Non-fusible fraction: `investigation_results/non_fusible_ops_accounting.csv`
- Oracle floors: `investigation_results/measured_oracle_floors.csv`
