# amax_sum_5f0c26b7e967

## Compile: 149.15us, Oracle: 100.03us, Gap: 1.49x

## Diagnosis: NEW_PATTERN

## Root cause: Inductor lowers the T5 causal relative-position attention softmax as a generic decomposed graph: iota/minimum/neg/log/bucket/embedding/permute/where/add/amax/sub/exp/sum/div/cast/expand/view, scheduling the bias computation and causal masking as separate pointwise producers feeding a reduction kernel. The oracle recomputes the logarithmic relative-position bucket lookup, embedding bias, and causal -65504 mask inline within one row-softmax Triton kernel, avoiding materializing the [2048,2048] intermediate bias/mask tensors.

## Fix path: Add an Inductor lowering that recognizes T5-style relative-position bucket computation and causal masking as cheap structured index computations that can be recomputed at point of use within the row-softmax kernel, rather than materialized as separate pointwise producers.

## Status: design_todo

## Details

- Model: torchbench_hf_T5 (inference)
- Pattern: amax+sum reduction (attention softmax with relative position bias)
- Inductor kernels: 1 unique Triton kernel (but with suboptimal fusion of the bias computation)
- Ops: iota, unsqueeze, add, sub, minimum, neg, lt, convert_element_type, div, log, mul, where, embedding, permute, amax([-1],keepdim), exp, sum([-1],keepdim), div, cast(f16), expand, view
- Shapes: [8,2048,2048] f16 input (bmm scores), [32,8] f16 bias table, output [8,2048,2048] f16
- Row length: 2048 (persistent kernel with full row in registers)
- Data volume: the 8*2048*2048 = 32M elements per tensor, with bias table only [32,8] = 256 elements
- The gap (1.49x) comes from: (1) Inductor materializing the [2048,2048] bias+mask intermediate (16MB in f16), (2) not recognizing that the bucket computation is O(1) per element and cheaper to recompute than to load from memory
- combo_kernels=True makes it worse (177us vs 149us)
- The oracle's key insight: the relative-position bucket is a pure function of (query_idx, key_idx) that costs ~15 ALU ops per element vs a 2-byte memory load -- always worth recomputing
- No existing Inductor config flag addresses this gap
