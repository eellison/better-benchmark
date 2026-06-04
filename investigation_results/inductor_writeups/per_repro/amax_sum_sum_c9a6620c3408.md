# amax_sum_sum_c9a6620c3408

## Compile: 2648.9us, Oracle: 2843.6us, Gap: 0.93x

## Diagnosis: AT_FLOOR

## Root cause: Computes the full MobileBERT masked-LM forward slice, bias add, ignore-index cross-entropy mean, and materialized softmax output with one online row kernel plus one scalar reduction.

## Fix path: Add a semantic lowering for biased log_softmax plus gather plus masked mean that can share online softmax row accumulators with required sibling materialized outputs.

## Status: closed

## Details

- Model: hf_MobileBertForMaskedLM_train_000
- Pattern: amax, sum, sum, sum reduction (25 ops)
- Oracle: oracle_online_softmax.py
