# var_mean_var_mean_73e1a842318f - oracle_dual_bn_residual_cat

## Classification
SCHEDULER_FUSION

## Benchmark Results
- Oracle: 28.42 us (2 kernels)
- Inductor: 45.98 us (2 kernels)
- Ratio: 1.618x (largest gap in this sample)

## Config Exploration
| Config | Time (us) |
|--------|-----------|
| combo_kernels + CDT | 45.98 |
| combo + multi_kernel=3 + per_subkernel_blocks | 48.04 |
| combo + multi3 + cooperative_reductions | 48.79 |

No config change helps. multi_kernel and coop_reductions make it slightly worse.

## Root Cause

The repro (from timm ghostnet_100 train) computes a dual BN + residual cat:
- BN1: var_mean over [512, 80, 7, 7] + affine + running stats
- cat([add_315, bn1_output], dim=1) -> [512, 160, 7, 7]
- BN2: var_mean over [512, 160, 7, 7] + affine + running stats  
- Residual add: cat_result + bn2_output -> final [512, 160, 7, 7]

Inductor generates 2 kernels:
1. First reduction: BN1 var_mean + running stats + affine (80 channels)
2. Second reduction: BN2 var_mean fused with cat + residual add + running stats + affine (160 channels)

The oracle also uses 2 kernels but achieves 1.6x better performance. The issue is **tiling and memory access patterns**:

The oracle tiles by channel and processes all N*H*W elements per channel in a single thread block, enabling:
- Perfect L2 cache reuse for the per-channel parameters
- Coalesced memory access patterns
- Efficient warp-level reductions

Inductor's generated reduction kernel processes the data in a less cache-friendly order. The BN reduction over [N, C, H, W] with reduction dims=[0,2,3] requires careful tiling to avoid scattered memory accesses. Inductor's current reduction codegen iterates in a way that causes more L2 cache thrashing for the 80/160 channel case with spatial dims 7x7.

The specific issue: with spatial=7x7=49, N=512, elements per channel = 512*49 = 25088. The oracle tiles this into BLOCK_K chunks per channel, keeping perfect channel locality. Inductor's norm template doesn't optimize the iteration order for this specific shape regime.

## Kernel Count
- Oracle: 2 kernels
- Inductor: 2 kernels (same count, but slower per-kernel)

## Fix Assessment
**Design doc** - Tiling/iteration order improvement needed in norm reduction template.

### What's needed:
The norm-template reduction codegen should optimize its iteration order for the case where:
- Reduction is over [N, H, W] (dims 0,2,3) keeping C
- N*H*W (25088) is moderate, fitting in persistent reduction
- Channel count (80, 160) is moderate

The current codegen likely assigns thread blocks in a way that doesn't maximize L2 locality for the channel parameters and intermediate accumulations. The oracle's channel-tiled approach processes all 25088 elements for a channel contiguously within a thread block.

### Relevant files:
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - norm reduction iteration order
- `/tmp/pytorch-work/torch/_inductor/choices.py` - persistent vs split reduction decision
- `/tmp/pytorch-work/torch/_inductor/codegen/triton_utils.py` - tile size heuristics

### Additional factor:
The cat operation between BN1 and BN2 means BN2's input comes from two sources. The oracle processes both halves of the cat directly inside the BN2 kernel without materializing the cat buffer. Inductor's second kernel reads from the cat buffer's first half (add_315) and the BN1 output (second half), but may not tile this optimally.

### Difficulty: Medium-Hard
Improving the norm reduction's memory access pattern for this shape regime requires changes to how the codegen orders its loops. This is not a fusion issue (Inductor achieves the same kernel count) but rather a code quality issue in the generated Triton code.
