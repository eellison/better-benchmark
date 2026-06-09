# sum_sum_d73ef74614dc

## Compile: 464.8us, Oracle: 195.1us, Gap: 2.38x

## Diagnosis: SCATTER_REDUCE

## Root cause: Computes the full Qwen RMSNorm-backward-style row gather/mask/reduce and indexed scatter-add tuple with Triton, writing the required vocabulary-gradient output directly from source rows.

## Fix path: Add a structured row-index scatter-reduce lowering that fuses the RMSNorm row reduction, side weight-gradient reduction, and duplicate-index scatter-add into the materialized output.

## Status: implemented

## Details

- Model: vllm_Qwen_Qwen3-30B-A3B_001
- Pattern: sum, sum reduction (34 ops)
- Oracle: oracle_structured_scatter_reduce.py
