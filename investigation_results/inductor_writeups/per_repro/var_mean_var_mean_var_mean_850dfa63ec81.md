# var_mean_var_mean_var_mean_850dfa63ec81 - oracle_triple_bn_sum_relu


## Measured Timings
- Oracle: 72.45 us
- Compile (CDT): not available
- Ratio: N/A

## Classification
SCHEDULER_FUSION (multi-reduction tiling)

## Benchmark Results
- Oracle: 74.59 us (3 kernels)
- Inductor: 126.85 us (1 kernel)
- Ratio: 1.701x

## Config Exploration
| Config | Time (us) |
|--------|-----------|
| combo_kernels + CDT | 126.85 |
| combo + multi_kernel=3 + per_subkernel_blocks | 125.17 |

No config change helps materially. multi_kernel=3 does not improve this case.

## Root Cause

The repro (from timm RepVGG-A2 training) computes triple training-BatchNorm with summed ReLU:
- Three independent var_mean reductions over [128, 384, 14, 14] (dims [0,2,3])
- Six running-stat copy_ updates (3 running_mean + 3 running_var)
- Three affine transforms
- Element-wise sum of all three normalized branches + ReLU -> output [128, 384, 14, 14]
- Side outputs: 3 invstd vectors, 3 mean views

### Oracle approach (3 kernels):
1. **Partial stats kernel** (48 x 99 grid): Tiles by channel (BLOCK_C=8) and spatial blocks (BLOCK_M=256). Computes sum and sum_sq for all three inputs simultaneously in a single pass through the data. Each thread block processes 8 channels x 256 spatial elements for all three inputs.
2. **Finalize stats kernel** (12 blocks): Reduces partial sums to mean/var/invstd per channel, updates running stats. Tiny kernel.
3. **Triple BN sum ReLU epilogue** (3840 blocks): Pointwise kernel that loads stats, normalizes all three inputs, sums, and applies ReLU. BLOCK=256 per thread block.

### Inductor approach (1 kernel):
- Single combo reduction kernel: `xnumel=384, r0_numel=25088`
- Processes each channel sequentially: reads all 25088 elements for x0, x1, x2, computes all three var_mean reductions inline, then writes the affine-sum-ReLU output.
- This is a **persistent reduction** (all 25088 elements per channel in a loop within one thread block).

### Why the oracle is 1.7x faster:

1. **Memory bandwidth utilization**: The oracle's partial stats kernel reads all three 128x384x14x14 inputs in a tiled parallel fashion with maximum bandwidth (many thread blocks). Inductor's single kernel has only 384 thread blocks (one per channel), severely limiting parallelism and L2 bandwidth utilization.

2. **Two-pass vs single-pass for reductions**: The oracle separates the reduction (stats computation) from the output production (epilogue). This means the output epilogue kernel is a simple pointwise with no sequential dependencies -- it can saturate memory bandwidth. Inductor's fused kernel must serialize: first compute stats (sequential loop over 25088 elements), then write output (sequential loop over 25088 elements again), doubling the sequential work per thread block.

3. **Occupancy**: With 384 thread blocks and 148 SMs on B200, Inductor gets only 2-3 waves. The oracle's epilogue kernel launches 3840 blocks (26 waves), achieving much better occupancy and hiding memory latency.

4. **Data reuse**: The oracle reads each input twice (once for stats, once for epilogue), same as Inductor. But the oracle's reads are parallel across many blocks, while Inductor serializes them within each channel's thread block.

## Kernel Count
- Oracle: 3 kernels (parallel tiled stats + finalize + parallel epilogue)
- Inductor: 1 kernel (sequential per-channel persistent reduction + output)

## Fix Assessment
**Design doc** - Scheduler-level enhancement needed.

### What's needed:
The scheduler should recognize the pattern of multiple independent var_mean reductions over the same spatial dimensions feeding into a combined pointwise consumer. Instead of fusing everything into one persistent reduction kernel (which limits parallelism), it should:

1. Use a **multi-block reduction** strategy: split the per-channel reduction across multiple thread blocks (partial sums), then finalize
2. Separate the reduction phase from the epilogue phase to allow the output kernel to run with full parallelism
3. Alternatively, use **cooperative reductions** where multiple thread blocks contribute to the same channel's statistics

### Key insight:
Inductor correctly fuses everything into 1 kernel (no unnecessary materialization). But the single-kernel approach is suboptimal because the reduction dimension (25088) is large enough that splitting it across multiple thread blocks would improve bandwidth utilization. The oracle demonstrates that a 3-kernel approach with better parallelism beats the 1-kernel fused approach.

### Relevant files:
- `/tmp/pytorch-work/torch/_inductor/choices.py` - split_reduction heuristic for large reductions
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - fusion scoring for multi-reduction patterns
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - multi-block reduction codegen

### Difficulty: Hard
This requires the scheduler to reason about when a fused kernel with poor parallelism should be split into multiple kernels for better GPU utilization. This is fundamentally a "fusion is not always better" insight that requires cost-model awareness of the GPU's SM count and achievable bandwidth.
