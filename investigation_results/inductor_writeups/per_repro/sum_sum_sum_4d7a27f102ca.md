# sum_sum_sum_4d7a27f102ca

## Compile: 763.9us, Oracle: 277.5us, Gap: 2.75x

## Diagnosis: SCATTER_REDUCE

## Root cause: Fuses both Swin relative-position batch reductions, duplicate-index scatter-adds, and the softmax-backward side-output store into one Triton producer.

## Fix path: Add a structured relative-position scatter-reduce lowering that recognizes index_put(accumulate=True) buckets and emits the dependent full-tensor output from the same producer.

## Status: implemented

## Details

- Model: timm_swin_base_patch4_window7_224_train
- Pattern: sum, sum, sum reduction (19 ops)
- Oracle: oracle_structured_scatter_reduce.py
