# pointwise_e5a8078b814d

## Compile: 341.82us, Oracle: 259.84us, Gap: 1.316x

## Classification: STENCIL_PRODUCER_INLINE

## Root Cause

The oracle fuses the complete BatchNorm affine + ReLU + 3x3 stride-2 max_pool stencil into one Triton kernel that writes only the final channels-last pooled tensor [128, 64, 73, 73]. Inductor materializes the full normalized/affine and ReLU tensor [128, 64, 147, 147] before the pooling stencil.

Inductor generates 2 kernels:
1. `triton_poi_fused_relu_0`: BN affine + ReLU, writing the full [128, 64, 147, 147] intermediate
2. `triton_poi_fused__low_memory_max_pool_with_offsets_relu_1`: max_pool reading the materialized intermediate

The oracle avoids the intermediate buffer entirely by computing BN+ReLU inline for each of the 9 stencil positions in the 3x3 max_pool window.

## Kernel Count
- Oracle: 1 kernel (BN + ReLU + 3x3 maxpool in one pass)
- Inductor: 2 kernels (BN+ReLU materialized, then maxpool reads it)

## Config Exploration
| Config | Result |
|--------|--------|
| combo_kernels + CDT | 341.82 us (1.316x) |
| multi_kernel=2 | not applicable (not a reduction) |
| multi_kernel=3 | not applicable (not a reduction) |

No config changes help. This is a structural scheduler limitation.

## Fix Assessment: Design doc

The scheduler cannot currently fuse a pointwise producer into a max_pool/avg_pool stencil consumer because `_low_memory_max_pool_with_offsets` uses shifted index access patterns that the fusion system does not support. The fix requires teaching the scheduler to:

1. Recognize that the pointwise producer's output only escapes through the pool consumer
2. Allow the pool kernel to recompute the producer for each stencil element
3. For 3x3 stride-2 maxpool, this means 9 recomputations of BN+ReLU per output element, but the memory savings from eliminating a 128*64*147*147*4 = ~702MB buffer round-trip far exceed the compute cost

### Memory analysis:
- Intermediate size: 128 * 64 * 147 * 147 * 4 bytes = ~702 MB (written then read)
- Oracle avoids this entirely by computing inline
- Net savings: ~1.4 GB of memory traffic eliminated

### Relevant files:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: can_fuse must handle stencil index patterns
- `/tmp/pytorch-work/torch/_inductor/lowering.py`: _low_memory_max_pool_with_offsets creates shifted loads
- `/tmp/pytorch-work/torch/_inductor/ir.py`: realize_hint blocks fusion before pool consumers

### Affected repro count:
This pattern (pointwise -> pool fusion) also affects pointwise_d89eb45ec337 (1.152x, avg_pool), pointwise_073278de7552 (1.166x, ReLU+maxpool+copy_), and likely others. The stencil-producer-inline enhancement would benefit multiple repros.

## Details
- Model: timm_adv_inception_v3 (infer)
- Shape: [128, 64, 147, 147] f32 channels-last -> max_pool(3x3, stride=2) -> [128, 64, 73, 73]
- Pattern: BN(mean, var, weight, bias) -> ReLU -> max_pool2d(3x3, stride=2)
- Input layout: channels-last NCHW
