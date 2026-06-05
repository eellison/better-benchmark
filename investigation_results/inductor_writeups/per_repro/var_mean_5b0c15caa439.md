# var_mean_5b0c15caa439

- **Gap**: 1.65x (oracle 28.70us, compile 47.39us)
- **Classification**: SCHEDULER_FUSION
- **Label**: torchbench_LearningToPaint_train_000
- **Pattern**: Training BatchNorm + residual + ReLU + avg_pool2d (full spatial extent)

## Computation structure

The repro implements a training-mode BatchNorm block from LearningToPaint:

```
Input: convolution_20 [1024, 512, 4, 4]  (32MB)
       relu_14        [1024, 512, 4, 4]  (32MB, residual)
       weight/bias    [512] each
       running_mean/var [512] each

Steps:
1. var_mean over [batch, H, W] dims -> mean [1,512,1,1], var [1,512,1,1]
2. rsqrt(var + eps) -> invstd
3. Update running_mean/var (copy_ side effects)
4. Normalize: (x - mean) * invstd * weight + bias
5. Residual add: normalized + relu_14
6. ReLU
7. avg_pool2d([4,4]) on the relu'd output -> [1024, 512, 1, 1]
8. View -> [1024, 512]

Outputs: invstd [512], pooled [1024,512], mean_view [1,512,1,1],
         updated running_mean [512], updated running_var [512]
```

## Inductor's generated code (2 kernels)

### Kernel 1: `triton_red_fused_add_copy__mul_relu_rsqrt_squeeze_sub_unsqueeze_var_mean_0`

- **Type**: Reduction (Welford), x_dim=512 (channels), r_dim=16384 (batch*HW)
- **First loop**: Welford accumulation over r_dim to compute mean and M2
- **Epilogue** (after reduction): Computes rsqrt, running stat updates (copy_)
- **Second loop**: Re-reads input, applies `(x - mean) * invstd * weight + bias + residual`,
  ReLU, and **stores the full [1024, 512, 4, 4] activated tensor** to `buf4` (32MB write)

### Kernel 2: `triton_poi_fused_add_avg_pool2d_mul_relu_sub_unsqueeze_1`

- **Type**: Pointwise, 524288 elements (1024*512)
- **Operation**: Reads 16 contiguous elements from `buf4` per output element, sums them,
  divides by 16 (avg_pool2d with full spatial extent 4x4 on 4x4 input)
- **Traffic**: 32MB read + 2MB write

## Oracle structure (3 kernels, less traffic)

1. **`_partial_stats_kernel`**: Grid = (n_blocks, channel_blocks). Reduces x into partial
   sums/squares per channel. Small intermediate (~32 partial blocks * 512 channels * 8 bytes).

2. **`_finalize_stats_kernel`**: Reduces partial sums -> mean/invstd, updates running stats.
   Negligible traffic.

3. **`_bn_residual_relu_pool_kernel`**: Grid = (1024, 8) with BLOCK_C=64, BLOCK_HW=16.
   Per (batch, channel_block): loads all 16 spatial elements of x and residual, applies
   `(x - mean) * invstd * weight + bias + residual`, ReLU, spatial sum (pool), writes
   only the pooled scalar. **No intermediate materialization.**

## Root cause

The performance gap comes from materializing a 32MB intermediate buffer (`buf4`) that is
written by Kernel 1's epilogue and read by Kernel 2. The oracle avoids this by having
a separate kernel (K3) that re-reads the original input and fuses
normalize + affine + residual + relu + spatial-pool into a single pass, writing only
the 2MB pooled output.

### Bandwidth analysis

| Path | Inductor | Oracle |
|------|----------|--------|
| Read x (stats) | 32MB | 32MB |
| Read x (normalize) | 32MB | 32MB |
| Read residual | 32MB | 32MB |
| Write intermediate | 32MB | 0 |
| Read intermediate | 32MB | 0 |
| Write pooled output | 2MB | 2MB |
| **Total** | **~162MB** | **~98MB** |

At ~3 TB/s bandwidth: extra traffic = 64MB, theoretical savings = ~21us.
Actual gap = ~19-35us (depending on measurement), consistent with bandwidth + kernel
launch overhead.

### Why Inductor cannot fuse today

The reduction kernel iterates per channel (x_dim=512) with r_dim = batch*HW = 16384.
The avg_pool needs output per (batch, channel) = 524288 elements. To fold the pool
into the reduction epilogue, each channel's thread would need 1024 batch-indexed
accumulators -- infeasible for the current codegen.

The alternative (oracle approach) requires the scheduler to:
1. Recognize that the reduction epilogue's only consumer is a spatial reduction (avg_pool)
2. NOT materialize the intermediate in the reduction epilogue
3. Generate a separate "apply" kernel with per-(batch, channel) tiling that re-reads
   the original input and the pre-computed statistics
4. Fuse normalize + affine + residual + relu + spatial-pool in that kernel

This is a multi-tiling scheduling decision that Inductor's current scheduler cannot express.

## Configs tested (none help)

| Config | Result |
|--------|--------|
| `combo_kernels=True` | No change (~75us) |
| `aggressive_fusion=True` | No change |
| `triton.nested_reduction=True` | No change |
| `_sibling_reduction_fusion=True` | No change |
| `triton.tile_reductions=True` | No change |
| `triton.cooperative_reductions=True` | No change |
| `epilogue_fusion=False` | Slightly worse (~78us) |
| `max_autotune=True` | No change |
| `split_reductions=False` | No change |

## Whether fixable without custom Triton kernels

**Not with current config options.** The fix requires a scheduler-level change:

### Proposed fix: Reduction-consumer re-tiling pass

When the scheduler detects that a reduction's epilogue produces a large intermediate
tensor whose only consumer is a "cheap reduction" (e.g., avg_pool2d over the same
spatial dims, or a sum over a small known dimension), it should:

1. Remove the intermediate materialization from the reduction epilogue
2. Emit a new fused kernel that reads the original inputs + the reduction's scalar
   outputs (mean, invstd) and applies the epilogue computation + consumer reduction
   with a different tiling (per batch*channel_block rather than per channel)

This is the same class of issue as the general "BN-training + downstream consumer"
fusion problem. It affects any pattern where:
- A reduction computes statistics
- An epilogue broadcasts those statistics back to the full tensor
- A downstream op immediately reduces that full tensor again (different dims)

## Relation to other repros

This is in the same family as other `var_mean` + downstream consumer patterns.
The distinctive element here is that the consumer is `avg_pool2d` with a full spatial
kernel (same size as input spatial dims), making it purely a spatial reduction. The
running-stat `copy_` side effects add complexity but are not the primary blocker.

## Status

- Classification: SCHEDULER_FUSION (confirmed)
- Oracle correctness: PASS
- Benchmark: oracle=28.70us, compile=47.39us, ratio=1.651x, status=GOOD
- Fix category: Scheduler redesign (reduction-consumer re-tiling)
- Fixable with config: No
- Concrete next step: Implement a scheduler pass that detects
  "reduction epilogue -> materialize large buffer -> small consumer reduction" and
  replaces it with "reduction -> emit separate fused apply+consumer kernel with
  different tiling"
