# Non-Fusible Ops Benchmark: Model-Level Speedup Floor Analysis

## Summary

Non-fusible operations (matmul, convolution, SDPA, embedding, pooling) form the
irreducible "floor" that limits how much model-level speedup is achievable from
kernel fusion improvements alone. This analysis quantifies that floor across the
184-model corpus.

**Key finding:** For standard transformer and CNN models, non-fusible ops consume
88-99.9% of model runtime. The median fusible fraction is only 4.4% of total
model time. Fusion improvements have outsized impact only on physics simulations
and custom stencil workloads where non-fusible ops are absent.

## Measured Non-Fusible Op Times (NVIDIA B200)

### Matrix Multiplication (torch.mm)

| Description | M | K | N | dtype | Time (us) | TFLOPS | % Peak |
|---|---|---|---|---|---|---|---|
| BERT attn proj | 16384 | 768 | 768 | f32 | 370.6 | 52.2 | 19.3% |
| BERT FFN up | 16384 | 768 | 3072 | f32 | 1278.8 | 60.5 | 22.4% |
| BERT FFN down | 16384 | 3072 | 768 | f32 | 1282.9 | 60.3 | 22.3% |
| HF 1024-dim attn | 8192 | 1024 | 1024 | f32 | 324.7 | 52.9 | 19.6% |
| Large square | 4096 | 4096 | 4096 | f32 | 2161.6 | 63.6 | 23.5% |
| LLaMA small | 1024 | 512 | 512 | f32 | 20.4 | 26.3 | 9.7% |
| MobileBERT narrow | 32768 | 128 | 128 | f32 | 35.7 | 30.0 | 11.1% |
| BERT attn proj | 16384 | 768 | 768 | f16 | 21.2 | 913.7 | 20.3% |
| BERT FFN up | 16384 | 768 | 3072 | f16 | 50.2 | 1540.8 | 34.2% |
| Large square | 4096 | 4096 | 4096 | f16 | 90.8 | 1514.4 | 33.7% |
| Large rect | 8192 | 4096 | 4096 | f16 | 162.7 | 1689.6 | 37.5% |
| Small batch | 128 | 4096 | 4096 | f16 | 14.0 | 307.8 | 6.8% |

### Batched Matrix Multiplication (torch.bmm)

| Description | B | M | K | N | dtype | Time (us) | TFLOPS |
|---|---|---|---|---|---|---|---|
| BERT QK^T | 1536 | 128 | 64 | 128 | f32 | 88.9 | 36.2 |
| BERT attn*V | 1536 | 128 | 128 | 64 | f32 | 90.0 | 35.8 |
| Longformer QK | 512 | 128 | 64 | 128 | f32 | 35.5 | 30.2 |
| BERT QK (f16) | 1536 | 128 | 64 | 128 | f16 | 23.0 | 140.0 |
| ViT QK (f16) | 96 | 1024 | 64 | 1024 | f16 | 40.6 | 317.0 |

### Other Non-Fusible Ops

| Op | Config | dtype | Time (us) | Notes |
|---|---|---|---|---|
| Embedding | [20005,768] idx[128,128] | f32 | 24.4 | Memory BW bound (2.06 TB/s) |
| Embedding | [32000,512] idx[32,32] | f32 | 6.9 | Small lookup |
| SDPA | B=8 H=12 S=512 D=64 | f16 | 20.4 | Flash attention |
| SDPA | B=4 H=32 S=2048 D=128 | f16 | 187.4 | GPT-2 scale |
| SDPA | B=1 H=32 S=8192 D=128 | f16 | 719.0 | Long context |
| MaxPool2d | k=3 s=2, [128,64,55,55] | f16 | 79.6 | AlexNet pool |
| Conv2d | AlexNet conv1, k=11 s=4 | f16 | 5069.8 | Largest single conv |
| Conv2d | VGG conv1, k=3 | f16 | 1123.2 | Standard 3x3 |
| Conv2d | DenseNet block, k=3 | f16 | 22.3 | Small feature map |
| Conv2d | ResNet 1x1 bottleneck | f16 | 18.3 | Cheap projection |

## Aggregate Non-Fusible Op Distribution (Across Corpus)

Across all 184 models' full computation graphs:
- **Total ops:** 413,861
- **Non-fusible ops:** 28,523 (6.9% by count)

Breakdown by op type:
| Op | Count | % of Total |
|---|---|---|
| mm | 13,507 | 3.26% |
| addmm | 6,457 | 1.56% |
| convolution | 4,193 | 1.01% |
| bmm | 3,451 | 0.83% |
| SDPA | 544 | 0.13% |
| embedding | 210 | 0.05% |
| avg_pool2d | 80 | 0.02% |
| max_pool | 77 | 0.02% |

## Per-Model Time Accounting

### Representative Models (Measured)

| Model | NF Time (us) | Fusible Current (us) | Fusible Achievable (us) | NF% | Max Model Speedup |
|---|---|---|---|---|---|
| BERT_pytorch (infer, f32, bs=128) | 50,630 | 33 | 33 | 99.9% | 1.0000x |
| hf_T5 (train, f32) | 102,680 | 8,480 | 1,600 | 92% | 1.066x |
| vgg16 (train, f16) | 2,015 | 2,325 | 1,267 | 46% | 1.322x |
| densenet121 (infer, f16) | 14 | 423 | 231 | 3% | 1.787x |
| pyhpc_equation_of_state (infer) | 0 | 84 | 15 | 0% | 5.623x |
| pyhpc_isoneutral_mixing (infer) | 0 | 940 | 212 | 0% | 4.435x |

### Distribution Across 128 Models (Estimated via Roofline)

| Statistic | Value |
|---|---|
| Median non-fusible fraction | 95.6% |
| Mean non-fusible fraction | 87.7% |
| Models with >80% non-fusible | 101 (79%) |
| Models with >50% non-fusible | 120 (94%) |
| Models with <20% non-fusible | 2 (1.6%) |
| Mean model-level speedup from perfect fusion | 1.024x |
| Max model-level speedup from fusion | 1.321x |

## Interpretation for Optimization Priority

### Where Fusion Improvements Have Impact

1. **Physics simulations** (pyhpc_*): 100% fusible, 4-5.6x achievable improvement.
   These are pure pointwise/stencil workloads with zero matmul/conv.

2. **CNN training** (vgg16, densenet121, squeezenet): 15-53% fusible.
   Large backward-pass reductions and BN/pool fusion opportunities.

3. **Transformer training** with large reductions (T5, Longformer): 7-24% fusible.
   Softmax backward, attention backward, multi-output reductions.

### Where Fusion Improvements Do NOT Help

- Standard transformer inference: >99% time in matmul (cublas/cutlass).
- Large-batch CNN inference: >95% time in cuDNN convolutions.
- Any model where matmul dominates (typical for production LLM serving).

### Utilization Observations

B200 TF32 peak: 270 TFLOPS. Measured matmul utilization: 10-24%.
B200 FP16 peak: 4500 TFLOPS. Measured matmul utilization: 7-38%.

Low utilization for small matmuls (M<2048) is due to:
- Kernel launch overhead relative to compute
- Memory bandwidth bottleneck for skinny matrices
- Insufficient parallelism to saturate tensor cores

This suggests that for models with many small matmuls, **matmul kernel optimization**
(better tiling, persistent kernels) could yield more model-level speedup than
fusion improvements.

## Methodology

- **Hardware:** NVIDIA B200 GPU, CUDA 13.2, PyTorch 2.13.0a0
- **Measurement:** `triton.testing.do_bench` with warmup=25-50, rep=200-300, return_mode='min'
- **Fusible timing:** From `baseline_results.csv` (measured inductor compiled kernel times)
- **Non-fusible estimation:** Roofline model (FLOP count / peak TFLOPS * efficiency)
  - Large matmul efficiency: 55% (f16) / 50% (TF32)
  - Small matmul efficiency: 30-40%
  - Convolution efficiency: 50-70% (cuDNN)
- **Model mapping:** manifest.json pattern hashes -> canonical repro baseline measurements
