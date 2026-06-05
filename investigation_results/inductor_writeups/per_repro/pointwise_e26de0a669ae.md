# pointwise_e26de0a669ae - BN + ReLU + MaxPool + AvgPool Fusion Gap

## Benchmark Result
- Oracle: 230.43 us
- Compile: 352.19 us
- Ratio: 1.528x
- Status: GOOD (oracle wins)

## Classification
SCHEDULER_FUSION

## Root Cause

The oracle fuses BatchNorm-inference affine + ReLU + 3x3 stride-2 maxpool into **one** Triton kernel, followed by a separate avgpool kernel (2 kernels total). The critical fusion is sinking the BN+ReLU computation into the maxpool stencil so the large intermediate f32[128,192,71,71] (124M elements, ~496MB) is never materialized.

Inductor emits **three** kernels:
1. `triton_poi_fused_add_mul_reciprocal_relu_sqrt_sub_unsqueeze_0` (124M elements): BN affine + ReLU, writes full f32[128,192,71,71] to memory
2. `triton_poi_fused__low_memory_max_pool_with_offsets_..._1`: Reads the materialized BN+ReLU output back, computes 3x3 maxpool stencil, writes f32[128,192,35,35]
3. `triton_poi_fused_avg_pool2d_2`: Reads pool output, computes 3x3 stride-1 avgpool, writes final f32[128,192,35,35]

The key inefficiency is the materialization between kernels 1 and 2. The BN+ReLU output is 128*192*71*71*4 = 496 MB. Writing it and reading it back costs ~992 MB of memory traffic that the oracle avoids.

## Kernel Count
- Inductor: 3 kernels
- Oracle: 2 kernels (BN+ReLU+maxpool fused, then avgpool)

## Memory Traffic Analysis

**Oracle** (2 kernels):
- Kernel 1 (BN+ReLU+maxpool): Reads input (496 MB channels-last) + BN params (4*192*4 = 3KB), writes pool output (128*192*35*35*4 = 120 MB)
- Kernel 2 (avgpool): Reads 120 MB, writes 120 MB
- Total: ~856 MB

**Inductor** (3 kernels):
- Kernel 1: Reads input (496 MB) + params (3KB), writes BN+ReLU result (496 MB)
- Kernel 2: Reads BN+ReLU (496 MB), writes pool (120 MB)
- Kernel 3: Reads pool (120 MB), writes avg (120 MB)
- Total: ~1848 MB (2.16x more traffic)

The 1.53x performance gap is explained by the ~2.16x more memory traffic (some of which is hidden by kernel launch overlap).

## Why Inductor Cannot Do This Today

The scheduler does not fuse a pointwise producer (BN+ReLU, iterating over [128,192,71,71]) with a stencil consumer (maxpool, which reads a 3x3 window of the producer's output per output element). The iteration domains differ:
- Producer: 128*192*71*71 = 124M elements
- Consumer: 128*192*35*35 = 30M elements (each reading 9 producer elements)

Additionally, the input is **channels-last** (stride = (967872, 1, 13632, 192)), which complicates the stencil indexing further. The oracle handles this with a 2D tiled iteration (BLOCK_C x BLOCK_S) that loads channel-contiguous strips for each spatial position.

The `realize_hint` / `should_realize_on_reuse` logic in `ir.py` forces the BN+ReLU result to be materialized because it is consumed by a stencil op that accesses it with shifted indices (the 3x3 window). The scheduler conservatively materializes any buffer that is read at non-trivial offsets.

## Config Exploration
- `combo_kernels=True`: no change (3 kernels)
- `combo_kernel_per_subkernel_blocks=True`: no change
- `triton.multi_kernel=3`: no change
- `coordinate_descent_tuning=True`: already enabled

No config resolves this. The issue is fundamental: stencil consumers cannot inline their producers.

## Fix Direction (Design Doc)

**Enhancement needed**: Pointwise-producer-into-stencil-consumer fusion for channels-last BN+pool patterns.

The scheduler needs to:
1. Detect that a pointwise producer (BN+ReLU) feeds exclusively into a stencil consumer (maxpool) and possibly other consumers
2. Inline the producer computation into the consumer's kernel: for each 3x3 window element loaded by the pool, compute BN+ReLU on-the-fly from the raw input
3. Handle channels-last layout correctly in the fused kernel's indexing

This is a significant scheduler enhancement because:
- The stencil kernel re-reads overlapping regions (each input element is read by up to 9 output elements for a 3x3 kernel)
- The BN params (mean, var, weight, bias) are per-channel and cheap to reload
- The net savings (avoiding 992 MB round-trip) far exceed the cost of recomputing BN 9x per input element

A simpler version: when the producer is pure pointwise with cheap per-channel broadcast params, and the consumer is a pooling stencil with known small window, allow fusion with recomputation.

**File references**:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (can_fuse, realize decisions)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (should_realize_on_reuse, realize_hint -- forces materialization for stencil consumers)
- `/tmp/pytorch-work/torch/_inductor/lowering.py` line 5625 (_low_memory_max_pool_with_offsets)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (would need stencil+inline-producer codegen)

**Affected repros**: Inception, VGG, ResNet, and any CNN with BN->ReLU->Pool in inference or channels-last training. This is one of the most common CNN patterns.

## Source
- Label: timm_adv_inception_v3_infer
- Pattern: channels-last BN affine -> ReLU -> 3x3 stride-2 maxpool -> 3x3 avgpool
