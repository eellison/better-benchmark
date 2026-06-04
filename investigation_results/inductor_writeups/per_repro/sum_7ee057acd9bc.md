# sum_7ee057acd9bc

## Compile: 357.3us, Oracle: 402.3us, Gap: 0.89x

## Diagnosis: AT_FLOOR

## Root cause: Fuses the max-pool offset decode, scatter-add overwrite mask, and final channel reduction into a direct gather-mask-reduce without materializing the dense scatter buffer.

## Fix path: Recognize max-pool backward scatter-add followed by pointwise masking and channel reduction as one structured scatter-reduce producer.

## Status: closed

## Details

- Model: torchbench_alexnet_train_001
- Pattern: sum reduction (8 ops)
- Oracle: oracle_structured_scatter_reduce.py
