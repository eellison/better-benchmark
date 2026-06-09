# sum_a3c9eb6b27f1

## Classification: SCATTER_REDUCE

## Oracle: oracle_rope_scatter_reduce.py

## Model: vllm_meta-llama_Llama-3.2-1B_001

## Measurements

- Compiled (cd): 29.3 us
- Oracle (triton): 50.2 us

## Diagnosis

Compile outperforms the oracle by 1.71x. The oracle applies the Llama RoPE grouped-head sum,
half-rotation slice-scatter, and transposed stores in one pass. Despite this structural fusion,
the oracle at 50.2us is significantly slower than Inductor's 29.3us generic lowering.

This is a complex 57-op graph with rotary embeddings (sin/cos, slice, slice_scatter, permute,
clone). The oracle's single-kernel approach for such a complex pattern apparently cannot match
Inductor's multi-kernel pipeline which benefits from better occupancy and simpler per-kernel
register pressure.

## Status

AT_FLOOR -- compile outperforms oracle by 1.71x. Oracle needs tuning but no Inductor work needed.

## Done Criteria

Closed. Compiled output already faster than oracle.
