# sum_sum_3219a09ab96a

## Compile: 605.1us, Oracle: 214.9us, Gap: 2.82x

## Diagnosis: SCATTER_REDUCE

## Root cause: Computes the full max-pool-backward scatter_add plus two masked channel sums as a direct gather-mask-reduce over the pool offsets, including the nonzero full-value mask contribution.

## Fix path: Recognize low-memory max-pool offset scatter-add feeding channel slices, where masks, and sibling reductions as one structured scatter-reduce producer instead of materializing the dense scatter buffer.

## Status: implemented

## Details

- Model: torchbench_squeezenet1_1_train_001
- Pattern: sum, sum reduction (12 ops)
- Oracle: oracle_structured_scatter_reduce.py
