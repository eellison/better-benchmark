# pointwise_7f55bac8afd0

## Compile: 1082.3us, Oracle: 836.5us, Gap: 1.29x

## Diagnosis: SCHEDULER_FUSION

## Root cause: Computes the ReLU max-pool values, pool offsets, and full input-shaped ReLU mask in one Triton stencil pass.

## Fix path: Fuse stencil consumers with same-producer layout side outputs by assigning ownership of input-layout writes to stencil tiles.

## Status: implemented

## Details

- Model: torchbench_squeezenet1_1_train_000
- Pattern: pointwise reduction (5 ops)
- Oracle: oracle_layout_stencil.py
