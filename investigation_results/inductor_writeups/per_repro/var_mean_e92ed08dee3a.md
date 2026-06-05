# var_mean_e92ed08dee3a

## Compile: 26.46us, Oracle: 22.27us, Gap: 1.188x

## Diagnosis: STENCIL_PRODUCER_INLINE

## Root cause: The oracle computes the complete VoVNet training BN var_mean, running-stat copy_ updates, affine ReLU, and low-memory 3x3 stride-2 ceil-mode maxpool-with-offsets scope in a fused manner. The BN normalization kernel computes stats and produces the normalized ReLU activation, then the maxpool stencil reads directly from that kernel's output without materializing the full activation to global memory.

Inductor generates 2 kernels: (1) BN-training reduction (var_mean + running-stat copy_ + rsqrt + affine + ReLU output), (2) low_memory_max_pool_with_offsets pointwise that re-reads the materialized [N,C,H,W] ReLU activation. The gap comes from the extra read of the full activation buffer by the pooling kernel.

## Kernel count
- Inductor: 2 kernels (red_fused_add_copy__mul_relu_rsqrt_squeeze_sub_unsqueeze_var_mean, poi_fused__low_memory_max_pool_with_offsets_add_mul_relu_rsqrt_sub_unsqueeze_var_mean)
- Oracle: 1 or 2 kernels (fused BN-train + stencil maxpool)

## Config exploration results
- multi_kernel=1 (default): 26.46us (ratio 1.188x)
- multi_kernel=2: 26.56us (ratio 1.184x) - no improvement
- multi_kernel=3: 26.53us (ratio 1.188x) - no improvement
- coordinate_descent_tuning=True, combo_kernels=True: already enabled

## Classification: STENCIL_PRODUCER_INLINE

The pointwise producer (BN-ReLU) cannot be inlined into the pool/stencil consumer because maxpool uses shifted indices. Inductor's scheduler cannot fuse a reduction producer (with mutation side outputs) into a multi-output low-memory maxpool consumer while preserving ceil-mode offset semantics.

## Fix path
Enhancement in `/tmp/pytorch-work/torch/_inductor/scheduler.py`: Teach the scheduler a training-BN-affine-ReLU-to-low-memory-maxpool template that:
1. Computes per-channel stats in a first pass
2. Reads the raw input, normalizes, applies ReLU, and performs the stencil maxpool in a second pass
3. Writes only the pooled output + offsets (not the intermediate activation)

This avoids materializing the full activation buffer between BN and maxpool.

## Status: design_doc
