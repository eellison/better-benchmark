# sum_sum_sum_21910bdd88af

## Summary

- Model: GPT2 (embedding backward + layernorm backward)
- Oracle: `oracle_gpt2_embedding_scatter_reduce.py`
- Classification: SCATTER_REDUCE
- Ratio: 1.173x (oracle 76.6us, compile 89.9us)
- Kernel count: Inductor 5 kernels, Oracle fewer (fused scatter-reduce)

## Root Cause

The repro computes the full GPT2 layernorm-backward/dropout return tuple by fusing shared rowwise hidden reductions with both duplicate-index position and vocabulary scatter-add outputs. Outputs:
- [768] channel sums (x2)
- [1024, 768] position embedding scatter-add
- [50257, 768] vocabulary embedding scatter-add

The oracle computes each token-row producer once and directly feeds:
1. Hidden (column) reductions for the two [768] gradient vectors
2. Indexed scatter-accumulation into the [1024, 768] position embedding gradient
3. Indexed scatter-accumulation into the [50257, 768] vocabulary embedding gradient

Inductor materializes the full `[8, 1024, 768]` producer intermediate and schedules:
1. Pointwise epilogue (layernorm backward computation)
2. Column reduction for the two [768] outputs
3. Separate index_put(accumulate=True) for position embeddings
4. Separate index_put(accumulate=True) for vocabulary embeddings

The gap comes from materializing the [8, 1024, 768] intermediate (24MB at fp32) and reading it multiple times.

## Config Exploration

| Config | Time (us) |
|--------|-----------|
| combo_kernels + CDT (default) | 89.9 |
| multi_kernel=2 | 91.9 |
| multi_kernel=3 | 97.3 |

No config closes the gap. The structural issue is the scatter pattern, not reduction strategy.

## Fix Assessment

**Design doc** -- requires embedding-backward scatter-reduce fusion.

### What's needed:
A scheduler enhancement that recognizes:
- Producer: pointwise computation over [B, seq_len, hidden]
- Consumers: column reductions + index_put(accumulate=True) with token indices
- These can share a single pass over the producer without materializing it

The embedding backward is a very common pattern (every transformer training run). An embedding-backward-specific lowering could:
1. Compute the layernorm backward per-row
2. Accumulate column sums for weight gradients
3. Scatter-add into embedding tables using token indices
All in one pass.

### Difficulty: Medium-High
The scatter indices (token IDs) create non-deterministic write conflicts requiring atomics. The challenge is coordinating the atomic scatter with the deterministic column reduction.

## Relevant Files

- `/tmp/pytorch-work/torch/_inductor/lowering.py`: index_put with accumulate
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: scatter + reduction fusion
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py`: embedding backward pattern

## Affected Repro Count

GPT2/DistillGPT2/GPT2-large embedding backward pattern. Multiple repros in the corpus use this same SCATTER_REDUCE pattern for embedding gradients.
