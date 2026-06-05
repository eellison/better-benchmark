# pointwise_d89eb45ec337

## Compile: 62.21us, Oracle: 54.02us, Gap: 1.152x

## Classification: STENCIL_PRODUCER_INLINE

## Root Cause

The oracle fuses the complete NFNet gated GELU + avg_pool2d into a single output-stencil kernel. It computes:
1. Sigmoid of [128,1536,1,1] SE channel attention (broadcast)
2. Multiply with [128,1536,12,12] feature map
3. Scale by 2.0 * scalar * 0.2
4. Add residual [128,1536,12,12]
5. Cast to f32, compute GELU (0.5 * x * (1 + erf(x * 0.7071)))
6. Cast back to f16, scale by NFNet factors
7. 2x2 stride-2 avg_pool2d -> [128,1536,6,6] f16 output

All in one kernel writing only the final pooled output.

Inductor generates 3 kernels:
1. `triton_poi_fused_sigmoid_0`: sigmoid of the [128,1536,1,1] SE attention
2. `triton_poi_fused_add_convert_element_type_erf_mul_sigmoid_1`: full pointwise chain producing [128,1536,12,12]
3. `triton_poi_fused_add_avg_pool2d_convert_element_type_erf_mul_sigmoid_2`: avg_pool2d reading the materialized intermediate

The fundamental issue is that Inductor's scheduler does not fuse a pointwise producer through an avg_pool2d stencil consumer. The avg_pool2d reads 4 neighboring values (2x2 window), requiring the full [128,1536,12,12] intermediate to be materialized before pooling.

## Kernel Count
- Oracle: 1 kernel (sigmoid + gated GELU + avgpool all fused)
- Inductor: 3 kernels (sigmoid, pointwise chain, avgpool)

## Config Exploration
| Config | Result |
|--------|--------|
| combo_kernels + CDT | 62.21 us (1.152x) |
| multi_kernel=2 | does not help (not a reduction pattern) |
| multi_kernel=3 | does not help (not a reduction pattern) |

No config change resolves this -- it requires a scheduler enhancement.

## Fix Assessment: Design doc

The scheduler needs to learn to fuse fixed-window pooling consumers (avg_pool2d, max_pool2d with small windows) with their pointwise producers. Instead of materializing the full intermediate, the kernel should compute the pointwise expression on-the-fly for each stencil element in the pooling window.

For a 2x2 avg_pool2d, this means computing the producer 4 times per output element but avoiding the full buffer materialization. The trade-off is compute vs memory bandwidth -- for this shape ([128,1536,12,12] -> [128,1536,6,6]), the memory savings from eliminating a 128*1536*12*12*2 = 56MB intermediate clearly wins.

### Relevant files:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: can_fuse needs to allow stencil consumers to absorb pointwise producers
- `/tmp/pytorch-work/torch/_inductor/lowering.py`: avg_pool2d lowering creates the structured access pattern
- `/tmp/pytorch-work/torch/_inductor/ir.py`: realize_hint on pool inputs blocks fusion

### Affected repro count:
This pattern (pointwise -> pool) affects multiple repros including pointwise_e5a8078b814d (BN+ReLU+maxpool, 1.316x gap) and likely others in the corpus.

## Details
- Model: torchbench_timm_nfnet (infer)
- Shape: [128, 1536, 12, 12] f16 input -> [128, 1536, 6, 6] f16 output
- Pattern: SE-sigmoid * conv + residual -> GELU -> NFNet scale -> avg_pool2d(2x2, stride=2)
- Memory traffic: oracle reads inputs once (~56MB), writes final output (~14MB). Inductor reads inputs, writes 56MB intermediate, reads it again for pool, writes 14MB final.
