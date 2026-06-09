# sum_sum_sum_8fe65349c733

## Compile: 706.7us, Oracle: 420.8us, Gap: 1.68x

## Diagnosis: SCATTER_REDUCE

## Root cause: Computes the full DenseNet backward tail with Triton reductions and a direct max-pool-backward gather, avoiding the dense zero/scatter_add materialization before the second BN-backward reduction and epilogue.

## Fix path: Teach the scheduler/codegen to lower structured max-pool-backward scatters as scatter-reduce producers that can feed sibling reductions and dependent BN epilogues without materializing the scatter buffer.

## Status: implemented

## Details

- Model: torchbench_densenet121_train_001
- Pattern: sum x4 reduction (81 ops)
- Oracle: oracle_multi_output_reduction.py
