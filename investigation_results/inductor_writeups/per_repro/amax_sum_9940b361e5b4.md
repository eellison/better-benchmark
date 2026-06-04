# amax_sum_9940b361e5b4

## Compile: 322.4us, Oracle: 174.1us, Gap: 1.85x

## Diagnosis: NEW_PATTERN

## Root cause: Computes the full Longformer sliding-window attention assembly, key/query masking, online row softmax, and final padded output layout directly in Triton.

## Fix path: Add a Longformer sliding-window attention pattern lowering that fuses the structured band assembly with the softmax reduction epilogue and destination-layout scatter.

## Status: implemented

## Details

- Model: torchbench_hf_Longformer_infer_002
- Pattern: amax, sum reduction (236 ops)
- Oracle: oracle_online_softmax.py
