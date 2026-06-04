# pointwise_a14dcfc06344

## Compile: 353.1us, Oracle: 189.6us, Gap: 1.86x

## Diagnosis: SCHEDULER_FUSION

## Root cause: Fuses each branch's BN-affine + ReLU producer directly into the 3x3 avg-pool stencil while writing the final concatenated channel layout.

## Fix path: Teach the scheduler to fuse layout-producing cat operands into same-channel stencil consumers and assign final-layout writes to the stencil tiles.

## Status: implemented

## Details

- Model: timm_adv_inception_v3_infer (6 shapes)
- Pattern: pointwise reduction (70 ops)
- Oracle: oracle_layout_stencil.py
