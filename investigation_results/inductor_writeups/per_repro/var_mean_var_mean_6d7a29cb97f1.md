# var_mean_var_mean_6d7a29cb97f1

## Compile: 443.5us, Oracle: 165.3us, Gap: 2.68x

## Diagnosis: RECOMPUTE_FUSION

## Root cause: Computes the full two-BN DenseNet stem region with shape-specialized Triton reductions, a fused first BN-affine-ReLU plus low-memory max-pool kernel, and a fused second BN epilogue.

## Fix path: Add a recompute-fusion path that sinks cheap BN-affine-ReLU pointwise work into overlapping pooling stencils instead of materializing the producer tensor.

## Status: implemented

## Details

- Model: torchbench_densenet121_train_000
- Pattern: var_mean, var_mean reduction (58 ops)
- Oracle: oracle_norm_template.py
