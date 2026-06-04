# sum_sum_2a2b5ec72fea

## Compile: 870.1us, Oracle: 570.2us, Gap: 1.53x

## Diagnosis: SCATTER_REDUCE

## Root cause: Computes the full cropped sigmoid-BN backward tuple with Triton channel reductions, gathering the structured 113x113->112x112 crop at the point of use instead of materializing the pad/crop and reduction producers.

## Fix path: Add a structured crop/pool-backward scatter-reduce lowering that can feed channel reductions and the dependent full-tensor BN epilogue from the same producer without generic intermediate tensors.

## Status: implemented

## Details

- Model: timm_tf_efficientnet_b0_train (4 shapes)
- Pattern: sum, sum reduction (47 ops)
- Oracle: oracle_structured_scatter_reduce.py
