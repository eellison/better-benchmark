# sum_9ba8ffa3e0fb

## Compile: 1146.0us, Oracle: 959.2us, Gap: 1.19x

## Diagnosis: SCATTER_REDUCE

## Root cause: Computes the complete Longformer structured pool/upsample backward graph with a row-sum prepass and direct final-layout scatter/gather materialization.

## Fix path: Recognize the structured Longformer slice-scatter/reduce/reshape pattern and lower it as one output-centric scatter-reduce producer with fused epilogues.

## Status: implemented

## Details

- Model: hf_AllenaiLongformerBase_train_005
- Pattern: sum reduction (102 ops)
- Oracle: oracle_structured_scatter_reduce.py
