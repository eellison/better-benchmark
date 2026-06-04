# amax_sum_any_d4c975ea57c8

## Compile: 71.9us, Oracle: 52.6us, Gap: 1.37x

## Diagnosis: NEW_PATTERN

## Root cause: Inductor lowers the additive-bias attention softmax with all-masked-row guard, dropout, and trailing layout transpose as separate generic reductions, pointwise RNG/dropout, and layout work over materialized intermediates, rather than recognizing this as a fused attention-softmax-dropout pattern with row-guard and emitting a single persistent kernel.

## Fix path: Add an Inductor pattern-match lowering for additive-bias attention softmax/dropout that preserves the all-masked-row guard (any/where) and fuses dropout plus the output-layout epilogue into the row softmax kernel.

## Status: design_todo

## Details

- Model: hf_MobileBertForMaskedLM (train, 1 shape)
- Pattern: amax+sum+any reduction (attention softmax with dropout and row-mask guard)
- Ops: view, add, amax, sub, exp, sum, div, eq, logical_not, any, where, inductor_random, mul, permute, expand
- Shape: [256,4,128,128] softmax -> [1024,128,128] transposed output with dropout
- The 1.37x gap comes from materializing exp/div intermediate and launching separate dropout + layout kernels.
