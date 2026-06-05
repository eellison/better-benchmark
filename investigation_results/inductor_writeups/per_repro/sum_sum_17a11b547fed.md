# sum_sum_17a11b547fed

## Compile: 235.42us, Oracle: 166.69us, Gap: 1.41x

## Diagnosis: SCATTER_REDUCE

## Root cause: Inductor schedules the Mistral-7B RMSNorm-backward plus indexed-accumulation scope as 5 separate Triton kernels: row-reduction for hidden-dimension dot products, a pointwise kernel for the RMSNorm gradient, a column-reduction for the [4096] weight gradient, a full f32 scatter-add (index_put with accumulate=True) over a [32768,4096] buffer, and a final bf16 cast kernel. The oracle shares Triton row-dot products, reduces the bf16 weight gradient, directly stores unique indexed rows, and only accumulates duplicate destination rows in a compact f32 buffer before the final bf16 [32768,4096] output.

## Fix path: Add a row-index scatter-reduce template that folds sentinel masking, duplicate accumulation, and final bf16 conversion into codegen while sharing the RMSNorm-backward row summaries. The key optimization is recognizing that most index rows are unique (only duplicates need accumulation) and that the scatter can be fused with the RMSNorm epilogue.

## Status: design_todo

## Details

- Model: vllm_mistralai_Mistral-7B-Instruct-v0.3 (inference backward)
- Pattern: sum+sum reduction (RMSNorm backward + scatter_reduce)
- Inductor kernels: 5 unique Triton kernels
- Ops: view, add, mul, convert_element_type (bf16<->f32), sum([2],keepdim), pow, div, expand, sum([0,1],keepdim), eq, unsqueeze, where, full, index_put(accumulate=True)
- Shapes: [2048,4096] bf16 inputs, [4,512,4096] intermediate, [4096] weight gradient, [32768,4096] scatter output
- Data volume: scatter output alone is 32768*4096*4 = 512MB in f32, dominates bandwidth
- The gap (1.41x) comes from: (1) materializing the full [32768,4096] f32 scatter buffer when most rows are written once, (2) separate bf16 cast pass over the scatter output, (3) redundant reads for the column reduction
- The oracle's key insight: partition rows into unique (direct bf16 store) and duplicate (compact f32 accumulate then cast), avoiding the full 512MB f32 buffer
- No existing Inductor config flag addresses this gap
