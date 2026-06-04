# sum_sum_145a8857ebf2

## Compile: 664.6us, Oracle: 398.4us, Gap: 1.67x

## Diagnosis: SCATTER_REDUCE

## Root cause: Computes the full max-pool-backward scatter feeding BN backward as a structured gather-mask-reduce, deriving channel sums directly from pool offsets and materializing only the required final output.

## Fix path: Recognize low-memory max-pool offset scatter-add plus pointwise mask plus sibling channel reductions as one structured scatter-reduce producer with a fused final-output epilogue.

## Status: implemented

## Details

- Model: torchbench_resnet50_train_001
- Pattern: sum, sum reduction (46 ops)
- Oracle: oracle_structured_scatter_reduce.py
