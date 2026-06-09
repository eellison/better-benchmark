# amax_sum_sum_031b4da8fbb0

## Compile: 20.38us, Oracle: 11.14us, Gap: 1.83x

## Diagnosis: SCHEDULER_FUSION

## Root cause: Inductor decomposes the ResNeSt split-attention forward scope into a single generic Triton kernel that handles the radix-2 softmax, fp16 cast, weighted branch sum, and avg_pool2d as separate scheduled regions. The oracle fuses all of these into one output-tiled kernel that: (1) computes fp32 softmax weights over the radix dimension inline, (2) rounds to fp16, (3) applies the weighted branch sum, and (4) directly accumulates the 3x3 stride-2 avg_pool2d stencil into the final [32,512,7,7] output tile. This eliminates the intermediate [32,512,14,14] materialization between the branch sum and pooling.

## Fix path: Extend Inductor's reduction/stencil fusion so that radix softmax weights and weighted branch sums can be sunk into a downstream avg_pool2d output loop. The scheduler needs to recognize that a small reduction (radix=2) feeding a broadcast multiply feeding a branch sum feeding a pooling stencil can all be computed within one output tile.

## Status: design_todo

## Details

- Model: torchbench_timm_resnest (inference)
- Pattern: amax+sum+sum reduction (split-attention softmax + weighted sum + avg_pool2d)
- Inductor kernels: 1 unique Triton kernel
- Ops: view, permute, convert_element_type (f16->f32), amax([1],keepdim), sub, exp, sum([1],keepdim), div, cast(f16), mul, sum([1]), avg_pool2d(3x3, stride=2, pad=1)
- Shapes: [32,1024,1,1] -> [32,2,512] softmax, [32,2,512,14,14] weighted branch sum, [32,512,14,14] -> [32,512,7,7] avg_pool2d
- The gap (1.83x) is the largest in this batch, driven by: (1) materializing the [32,512,14,14] intermediate between sum and pool (32*512*14*14*2 = 12.5MB), (2) separate load/store for the pooling pass
- combo_kernels=True makes it worse (24.35us vs 20.38us)
- The oracle achieves this by computing each 7x7 output pixel as a 3x3 stencil over the branch-sum result, with the branch weights computed inline from the softmax
- No existing Inductor config flag addresses this gap
