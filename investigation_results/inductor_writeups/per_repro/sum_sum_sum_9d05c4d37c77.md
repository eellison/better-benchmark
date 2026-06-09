# sum_sum_sum_9d05c4d37c77

## Compile: 704.3us, Oracle: 706.3us, Gap: 1.00x

## Diagnosis: AT_FLOOR

## Root cause: Computes the full avg-pool-backward/add producer directly inside the sibling BN/ReLU-backward reductions and recomputes it in the dependent tensor epilogue, avoiding materialized slice/where buffers.

## Fix path: Add scheduler support for recompute fusion through shared stencil producers into multi-output channel reductions and their post-reduction epilogues.

## Status: closed

## Details

- Model: timm_adv_inception_v3_train (6 shapes)
- Pattern: sum x8 reduction (164 ops)
- Oracle: oracle_multi_output_reduction.py
