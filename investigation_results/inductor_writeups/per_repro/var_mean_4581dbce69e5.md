# var_mean_4581dbce69e5

## Compile: 22.08us, Oracle: 15.23us, Gap: 1.45x

## Diagnosis: SCHEDULER_FUSION

## Root cause: Inductor emits 1 kernel that fuses the cat, var_mean reduction, running stat updates, affine+ReLU epilogue into a single `triton_red_fused_add_cat_copy__mul_relu_rsqrt_squeeze_sub_unsqueeze_var_mean` reduction kernel. The oracle achieves 1.45x speedup by computing the batch-norm statistics directly from the 6 source tensors (avoiding the logical concatenation indexing) and using a channel-tiled reduction plan that processes the cat sources with explicit offset arithmetic rather than Inductor's generic cat indexing expression.

## Fix path: Optimize Inductor's cat-through-reduction pattern to avoid the per-element cat index computation inside the reduction loop. When a cat is immediately consumed by a reduction over the concatenated dimension's complementary axes, the scheduler could rewrite the inner function to directly read from each source tensor with computed offsets, eliminating the conditional branching in the cat indexer.

## Status: design_todo

## Details

- Model: torchbench_densenet121 training
- Pattern: cat([512,7,7], [32,7,7]*5, dim=1) -> var_mean(dims=[0,2,3]) -> BN affine+ReLU + running stat copy_
- Inductor kernel count: 1 (good fusion, suboptimal cat indexing)
- Shapes: 6 inputs cat'd to [64, 672, 7, 7], reduction over 3136 elements per channel
- Output: relu [64,672,7,7], rsqrt [672], mean [1,672,1,1], running_mean [672], running_var [672]
- coord_descent_tuning: 26.62us (worse), combo_kernels did not help
- The overhead comes from the cat indexing expression evaluated for each of the 3136 reduction elements per channel, which involves multiple conditional branches to select among 6 source tensors
- Oracle uses explicit channel-range checks to load directly from each source tensor without branching
