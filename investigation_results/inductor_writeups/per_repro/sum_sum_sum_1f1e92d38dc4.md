# sum_sum_sum_1f1e92d38dc4

## Classification: SCATTER_REDUCE

## Benchmark Results
- Oracle: 341.76 us
- Inductor compiled: 513.86 us
- Ratio: 1.504x

## Kernel Count
- Inductor: 7 kernels
- Oracle: 3 kernels (row stats + scatter + finalize)

## Root Cause

The oracle computes the complete DeBERTa dropout/add/layernorm-backward row scope, including:
1. Two channel reductions [1536] (LN weight/bias gradients)
2. A [512, 1536] accumulated index_put output (scatter-reduce for embedding gradient)
3. A [128100, 1536] base-plus-scatter output (full vocabulary gradient)

Shape info: BATCH=8, TOKENS=512, HIDDEN=1536, ROWS=4096, VOCAB_ROWS=128100.

The oracle strategy (3 kernels):
1. `_row_stats_kernel`: streams through the [4096, 1536] domain, computes LN-backward row statistics (mean, variance), accumulates partial channel sums for weight/bias gradients
2. Scatter kernel: clones the live base tensor and scatters updates with `accumulate=True` using pre-computed row data
3. Finalize: reduces partials to final [1536] vectors

Inductor's 7-kernel decomposition:
1. Dropout pointwise (bernoulli mask + scale)
2. Add + LN-backward row reductions
3. LN-backward epilogue (pointwise)
4. Channel reduction for weight grad: sum([0]) over [4096, 1536]
5. Channel reduction for bias grad: sum([0]) over [4096, 1536]
6. index_put with accumulate=True -> [512, 1536]
7. Base tensor add for [128100, 1536] output

The key issue: Inductor cannot represent `index_put(accumulate=True)` as a scatter-reduce node that shares reusable row reductions with sibling channel reductions. It materializes intermediate buffers between the LN-backward computation, the channel reductions, and the scatter operations. The scatter itself is lowered as a separate kernel with its own memory traffic.

## Config Exploration
- `combo_kernels=True, coordinate_descent_tuning=True`: 513.86 us (baseline)
- The gap is structural: requires fusing scatter-reduce with row reductions

## What Inductor Needs (Design Doc)

**Enhancement needed**: index_put scatter-reduce lowering that fuses with compatible row/channel reductions.

The enhancement would:
1. Recognize index_put(accumulate=True) as a scatter-reduce node in the IR
2. Allow the scheduler to fuse it with sibling reductions that share the same row data
3. Emit a combined kernel that:
   - Computes row-local LN-backward statistics
   - Accumulates channel partial sums
   - Performs the scatter-reduce directly (clone base + atomic add at indexed positions)
   - Handles the sentinel/where masking inline

**Files to modify**:
- `/tmp/pytorch-work/torch/_inductor/lowering.py` - scatter-reduce IR lowering for index_put(accumulate=True)
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - fusion of scatter nodes with sibling reductions
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - scatter-reduce template with atomic accumulation
- `/tmp/pytorch-work/torch/_inductor/ir.py` - ScatterReduce IR node type

**Affected repro count**: DeBERTa and other models with embedding gradient scatter + LN-backward in the same scope. Estimated 3-5 affected repros.
