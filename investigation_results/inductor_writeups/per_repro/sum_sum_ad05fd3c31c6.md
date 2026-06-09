# sum_sum_ad05fd3c31c6

## Compile: 149.41us, Oracle: 77.73us, Gap: 1.92x

## Diagnosis: SCATTER_REDUCE / SCHEDULER_FUSION

## Root cause: Inductor generates a single fused kernel but uses a 2-pass approach: the first pass computes the `where` values (ReLU-gated pool-backward gradient) and accumulates two channel reductions (sum_where and sum_centered), storing intermediate `where` values to global memory; the second pass re-reads those intermediates plus the original inputs to compute the final BN-backward output using the reduction results. The oracle (`oracle_structured_pool_upsample_backward_reduce`) fuses everything into a single per-channel kernel that computes both reductions and the output epilogue in one pass over the data, reading the large [512,960,7,7] tensor only once.

## Kernel count
- Inductor: 1 kernel (but 2 passes over 25088 elements per channel within the kernel)
- Oracle: 1 kernel (single pass per channel, 960 programs, BLOCK_SIZE=32768)

## Config exploration
- `coordinate_descent_tuning=True`: already enabled, no improvement
- `combo_kernels=True`: already enabled, no improvement
- The issue is not kernel count but data access pattern -- Inductor's kernel materializes the `where` intermediate to memory and re-reads it in the second pass

## Fix path: The scheduler needs awareness that when a reduction's result is consumed by a pointwise epilogue that also reads the reduction's input, the reduction and epilogue can be fused into a single pass per channel tile. Currently Inductor's `triton_red` template always materializes reduction intermediates to a buffer between the reduction loop and the epilogue loop. A "single-pass BN-backward" template that keeps both reduction accumulators and the per-element output computation in the same per-channel iteration would eliminate the extra memory round-trip.

## Status: design_todo

## Details

- Model: timm_ghostnet_100 (train)
- Pattern: sum+sum with ReLU-gated pool-backward scatter, then dependent full-tensor epilogue
- Ops: squeeze, as_strided_scatter, expand, div, sub, mul, relu, le, where, sum([0,2,3]) x2, sub, mul -> output tensor [512,960,7,7] + vector [960]
- Shape: [512,960,7,7] f32 (N=512, C=960, H=W=7)
- Reduction size per channel: N*H*W = 512*49 = 25088 elements
- Data volume: 512*960*49*4 = ~92MB per full tensor read
- The 1.92x gap comes from the second pass re-reading 92MB of intermediate data plus the original 92MB input tensor, totaling ~184MB of extra memory traffic compared to the oracle's single-pass approach
- File: `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (reduction template), `/tmp/pytorch-work/torch/_inductor/scheduler.py` (fusion decisions)
- The key Inductor limitation: the `triton_red` heuristic forces a 2-pass structure whenever the reduction result feeds back into a pointwise computation over the same spatial domain
