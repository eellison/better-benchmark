# var_mean_var_mean_var_mean_97062ec87370 - oracle_bn_relu_spatial_mean

## Classification
SCHEDULER_FUSION

## Benchmark Results
- Oracle: 69.63 us (2 kernels)
- Inductor: 93.09 us (3 kernels)
- Ratio: 1.337x

## Config Exploration
| Config | Time (us) |
|--------|-----------|
| combo_kernels + CDT | 93.09 |
| combo + multi_kernel=3 + per_subkernel_blocks | 119.14 |

multi_kernel=3 actually makes this worse (119 us vs 93 us), likely due to the combo kernel dispatch overhead.

## Root Cause

The repro (from timm adv_inception_v3 training) computes 6 branches of training-BatchNorm + affine ReLU, followed by channel concatenation and spatial mean:
- Six independent var_mean reductions over different input tensors:
  - 1x [128, 320, 8, 8]
  - 4x [128, 384, 8, 8]
  - 1x [128, 192, 8, 8]
- Six running-stat copy_ updates (12 total: 6 running_mean + 6 running_var)
- Six affine + ReLU epilogues
- Three cat operations (grouping branches into concatenated tensors)
- Final spatial mean over [128, 2048, 8, 8] -> [128, 2048] output

### Oracle approach (2 kernels):
1. **Stats kernel** (2048 thread blocks, 1 per output channel): Each block computes the per-channel var_mean stats by loading from the appropriate branch input based on channel offset, using the "virtual concat" pattern. Updates running stats inline. Uses BLOCK_K=8192 (persistent, fits all 128*8*8=8192 elements per channel).
2. **Spatial mean kernel** (128 x 128 grid): For each (batch, channel_block), re-reads the branch input, applies normalize + affine + ReLU, and computes spatial mean over 8x8=64 elements directly, writing the [128, 2048] output without materializing normalized activations or concatenated tensors.

The oracle never materializes:
- The normalized [128, C, 8, 8] activations (6 tensors, ~75 MB total)
- The concatenated [128, 2048, 8, 8] tensor (~128 MB)

### Inductor approach (3 kernels):
1. **Combo reduction kernel** (`triton_red_fused_0`): Uses `pid % 6` dispatch to handle all 6 branches in one kernel launch. Computes var_mean + running stats for all branches. Good -- similar to oracle's stats kernel.
2. **Combo pointwise kernel** (`triton_poi_fused_1`): Uses `pid % 2` dispatch to write the normalized + affine + ReLU activations to two different cat buffers (since there are intermediate cats). This MATERIALIZES the normalized activations.
3. **Persistent spatial mean** (`triton_per_fused_mean_2`): Reduces the materialized concatenated tensor [128, 2048, 8, 8] over the last two dims. xnumel=262144 (128*2048), r0_numel=64 (8*8).

### Why the oracle is 1.34x faster:

1. **Avoids materializing normalized activations**: The oracle's kernel 2 re-reads the raw inputs and recomputes the normalization inline while computing the spatial mean. This saves writing ~75 MB of normalized activations and reading them back for the spatial mean -- a round trip of ~150 MB.

2. **Avoids materializing the cat**: Inductor must write a contiguous [128, 2048, 8, 8] buffer for the spatial mean kernel to read. The oracle's spatial mean kernel reads directly from branch inputs at computed offsets.

3. **Memory traffic comparison**:
   - Oracle: reads inputs 2x (stats + spatial_mean), writes [128, 2048] output + running stats = ~(2 * 75 + 1) MB reads, ~1 MB writes = ~151 MB total
   - Inductor: reads inputs 1x for stats, writes normalized outputs, reads them for spatial_mean = ~75 + 75 + 75 + 1 = ~226 MB total
   - Savings: ~75 MB (one full set of normalized activations)

4. **The spatial mean is cheap**: With only 64 elements per channel to reduce, the spatial mean is trivially persistent. The oracle exploits this by fusing norm+relu+mean into a single kernel that reads raw data and writes final output.

## Kernel Count
- Oracle: 2 kernels (stats + fused norm-relu-mean)
- Inductor: 3 kernels (combo stats + combo normalize/cat + spatial mean)

## Fix Assessment
**Design doc** - Scheduler fusion enhancement needed.

### What's needed:
The scheduler should recognize that when a spatial mean (small reduction) consumes the output of BN normalize+ReLU, it is profitable to:
1. NOT materialize the normalized activations
2. Instead, fuse the normalize+ReLU into the spatial mean kernel, re-reading raw inputs and loading stats from the first kernel

This is a variant of the "recompute-fusion" pattern: the normalization is cheap (just pointwise ops on per-channel stats), and the downstream consumer (spatial mean over 64 elements) is tiny. The cost of recomputing normalization inside the mean kernel is negligible compared to the memory savings of not materializing the intermediate.

### Specific scheduler change:
In `scheduler.py`, when scoring fusion between:
- A pointwise producer that writes a large tensor (normalized activations)
- A small persistent reduction consumer (spatial mean over small spatial dims)

The scheduler should prefer to keep the producer virtual and let the consumer recompute it, IF:
- The producer's inputs are still available (not freed)
- The producer computation is cheap (pointwise only)
- The consumer's reduction dimension is small (persistent-eligible)

### Relevant files:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - fusion scoring, realize decisions
- `/tmp/pytorch-work/torch/_inductor/ir.py` - should_realize_on_reuse, realize_hint
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - persistent reduction with inline producers

### Additional insight:
The combo_kernels infrastructure already groups the 6 BN reductions into one launch (good). But the materialization of normalized activations before the spatial mean is the bottleneck. The fix is about teaching the scheduler that "normalize + small_mean" can be fused even when it crosses a reduction boundary, because the mean reduction is trivially persistent (64 elements).

### Difficulty: Medium
The scheduler needs a cost model that compares: materialization cost (write + read intermediate) vs recomputation cost (re-read inputs + recompute cheap pointwise). For this specific pattern (BN normalize feeding tiny spatial mean), recomputation always wins.
