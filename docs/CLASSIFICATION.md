# Multi-Kernel Inference Fusion Failure Classification

59 repros across 33 models (26 HF dynamo + 7 vLLM), all producing >1 kernel.

## Category 1: Cross-entropy loss — numel/rnumel mismatch (16 repros, all k=2)

**Pattern**: `amax(logits, dim=-1) → sub → exp → sum → log → sub → gather → neg → where → sum.default → div`

This is the standard `F.cross_entropy` / `loss_fct(predictions, labels)` pattern. The softmax
reductions (amax, sum over vocab dim) operate on shape `[batch*seq, vocab]` while the final
`sum.default` (scalar reduction of per-token losses) operates on `[batch*seq]`. The two reductions
have completely different (numel, rnumel): e.g., `(4096, 1)` vs `(30000, 4096)`.

**Root cause**: The per-token loss reduction `sum([batch*seq])` can't fuse with the softmax
reductions `amax/sum([batch*seq, vocab])` because their iteration+reduction domains differ entirely.

**Models**: Albert, Bart, Blenderbot, DebertaV2, GPT2, GPTNeo, Longformer, M2M100, MBart, MT5,
MegatronBert, OPT, Qwen3-0.6B, Reformer

**Fusion blocker**: `numel/rnumel mismatch (reduce)`

**Would benefit from**: Partitioned fusion (different sub-grids for different reduction shapes), or
just recognizing that the scalar sum is tiny and not worth a separate kernel.

---

## Category 2: Q/K RoPE + GQA expand — no shared data / different iteration domains (8 repros, k=3-4)

**Pattern**: `q_proj → reshape → permute → RoPE(q) → output_q` AND
`k_proj → reshape → permute → RoPE(k) → expand → clone → output_k`

Q has 32 heads, K has 8 heads. The RoPE computation (mul + pointwise_cat + mul + add) is identical
for both but operates on different iteration domains:
- Q RoPE: `[4, 32, 512, 128]` = 8M elements
- K RoPE: `[4, 8, 512, 128]` = 2M elements
- GQA expand: `[4, 8, 4, 512, 128]` = 8M elements (reads K RoPE output)

Both Q and K read the same cos/sin embeddings `[1, 512, 128]`, but the scheduler reports
"no shared data" because the MemoryDep indexing expressions differ (different head dimension
in the stride pattern).

**Root cause**: The `score_fusion_memory` check requires exact `MemoryDep` match (same buffer name
AND same index expression). Q reads cos as `MemoryDep('arg1_1', 128*d1 + d3, {d0: 4, d1: 512, d2: 32, d3: 128})`
while K reads it as `MemoryDep('arg1_1', 128*d1 + d3, {d0: 4, d1: 512, d2: 8, d3: 128})`.
The `d2` range differs so these are not equal Deps despite accessing identical memory.

**Models**: Mistral, DeepSeek-V3, Qwen3-30B, GPT-OSS (2 patterns), Qwen3-0.6B

**Fusion blocker**: Different iteration domains prevent grid sharing. Shared reads of cos/sin
not detected because MemoryDep equality requires matching var_ranges.

**Would benefit from**: Partitioned fusion that recognizes cos/sin are shared reads and can emit
a single kernel with multiple output regions. The GQA expand can inline K RoPE output instead of
materializing the intermediate.

### Variant: k=4 Mistral/Qwen with inline RoPE embedding computation

Same as above but the region also includes the `inv_freq @ position_ids → cos/sin` computation,
adding one more kernel for the position embedding generation.

---

## Category 3: Cross-entropy with two-stage reduction — both mismatch + no shared data (12 repros, all k=3)

**Pattern**: Same as Category 1 but the final `sum.default` gets split into two stages by inductor:
- Stage 1: `sum([batch*seq]) → [batch*seq/SPLIT]` (large rnumel)
- Stage 2: `sum([batch*seq/SPLIT]) → scalar` (small rnumel)

This creates 3 kernels: (1) softmax reductions, (2) per-token loss + stage-1 sum + count,
(3) stage-2 sum + division.

**Root cause**: Same as Category 1, plus the two-stage split introduces an additional
numel/rnumel mismatch between stages.

**Models**: Bert, DistilBert, DistillGPT2, Electra, GoogleFnet, LayoutLM, MobileBert, PLBart,
Pegasus, Reformer, Roberta

**Fusion blocker**: `numel/rnumel mismatch (reduce)` AND `no shared data`

---

## Category 4: MoE bounds-check assertions — numel/rnumel mismatch + assert_async (1 repro, k=4)

**Pattern**: `one_hot(routing, num_experts)` implemented as bounds checks via
`ge(0) → any() → logical_not → _assert_async` AND `lt(num_experts) → any() → ...`
plus the actual one_hot: `eq(iota) → sum → gt`.

The `any` reduction gets split into two stages: stage-1 over [8192] rnumel, stage-2 over [2].
The two-stage split creates numel/rnumel mismatch that prevents fusion.

**Models**: Qwen3-30B-A3B (MoE model)

**Fusion blocker**: `numel/rnumel mismatch (reduce)` between stage-1 and stage-2 of `any`.
Also `_assert_async` has side effects preventing fusion across it.

---

## Category 5: ConcatKernel / NopKernel fusion boundary (2 repros, k=3)

**Pattern**: DeepSeek-V3 and Qwen3-30B position_diff: `slice → sub → cat → slice → slice → sub → ne`

The `cat` is lowered as `ConcatKernel` (NopKernel) instead of `pointwise_cat`. The
`NopKernelSchedulerNode.can_fuse()` returns False, creating a hard fusion boundary.

**Root cause**: `pointwise_cat` was not chosen because the cat inputs have multi-consumers
(`any_input_has_multi_consumers()` returns True) — the expand of position_ids is used by
both the cat and the subsequent diff computation.

**Models**: DeepSeek-V3, Qwen3-30B

**Fusion blocker**: `NopKernelSchedulerNode` (ConcatKernel) blocks all fusion.

---

## Category 6: Longformer sliding window — no shared data (3 repros, k=2)

**Pattern**: Value projection + as_strided sliding window chunking + clone.

Longformer uses `as_strided` to create overlapping chunks for sliding window attention.
The value projection and the strided clone operate on completely independent data paths with
different shapes and no shared buffers.

**Models**: AllenaiLongformerBase (3 regions)

**Fusion blocker**: `no shared data` — genuinely independent computations.

---

## Category 7: Qwen3 RMSNorm + RoPE + mean — no shared data + intermediate nodes (5 repros, k=3-6)

**Pattern**: `q_proj → RMSNorm(pow→mean→rsqrt→mul) → RoPE` AND same for K, plus the mean
reductions introduce intermediate scheduling dependencies.

The mean reduction for RMSNorm variance creates an intermediate node that blocks fusion of the
subsequent mul/RoPE with the Q path. The "intermediate nodes between node1 & node2" error
means there's a scheduling dependency chain that prevents reordering.

**Models**: Qwen3-0.6B (3 patterns), Qwen3-30B-A3B (1 pattern), Qwen3-0.6B vllm (1 pattern)

**Fusion blocker**: `intermediate nodes between node1 & node2` + `no shared data`

**Note**: k=6 cases include the full RoPE embedding computation (inv_freq @ position_ids → cos/sin)
in addition to the Q/K RMSNorm + RoPE paths.

---

## Category 8: Large var_mean / embedding graphs — no fusion log reasons (12 repros, k=4-25)

**Pattern**: These are large subgraphs containing `var_mean` (layer norm) + embedding lookups +
position encoding + sometimes cross-entropy loss — all in one extracted region. The high kernel
count comes from many independent sub-computations that genuinely can't fuse.

The fusion logger shows "NO REASONS" because within each pair that's attempted, fusion succeeds.
The high kernel count comes from the sheer number of independent operation clusters.

**Models**: DistillGPT2, GPT2, PLBart, M2M100, Bart, MBart, Pegasus, DebertaV2, Blenderbot

**Note**: These are large extracted regions (>50 ops) that probably shouldn't have been extracted
as single regions. The multi-kernel output is expected behavior — many independent computations
in one graph break.

---

## Category 9: Reformer-specific patterns (4 repros, k=2-3)

**Pattern**: Various Reformer-specific ops: LSH hash vectors with argmax + argsort, chunked
attention reshaping, and reversible layer duplicate computations.

**Models**: Reformer (4 regions)

**Fusion blocker**: Mixed — some `no shared data`, some `numel/rnumel mismatch`, one silent.

---

## Category 10: GPT-OSS softmax with non-standard dim — numel incompatibility (1 repro, k=2)

**Pattern**: Softmax over dim=-1 on a `[4, 64, 512, 513]` tensor (512+1 for attention sink).
The subsequent slice+clone operates on `[4, 64, 512, 512]`.

**Models**: GPT-OSS

**Fusion blocker**: `nodes numel incompatibility` — the softmax output [4,64,512,513] and the
sliced result [4,64,512,512] have different numels.

---

## Summary by root cause

| Root Cause | Repros | Models | Kernel Range |
|---|---|---|---|
| Cross-entropy reduction mismatch | 28 | ~18 | k=2-3 |
| Q/K RoPE iteration domain mismatch | 8 | 6 | k=3-4 |
| Qwen3 RMSNorm+RoPE intermediate deps | 5 | 2 | k=3-6 |
| Large multi-op regions (expected) | 12 | 9 | k=4-25 |
| ConcatKernel fusion boundary | 2 | 2 | k=3 |
| Longformer sliding window (independent) | 3 | 1 | k=2 |
| Reformer-specific | 4 | 1 | k=2-3 |
| MoE assert_async + two-stage any | 1 | 1 | k=4 |
| GPT-OSS numel incompatibility | 1 | 1 | k=2 |

## Impact-weighted priorities (for vLLM inference)

1. **Q/K RoPE + GQA** — hits every single LLM with GQA (Mistral, Llama, Qwen, DeepSeek, etc.). 
   3-4 extra kernel launches per attention layer. Wastes bandwidth re-reading cos/sin and
   materializing K RoPE intermediate.

2. **ConcatKernel position_diff** — hits DeepSeek-V3, Qwen3 (packed sequence detection).
   3 kernels where 1 would suffice.

3. **Qwen3 RMSNorm+RoPE** — hits Qwen3 family. RMSNorm mean creates scheduling barrier
   that prevents fusing the subsequent RoPE.

4. **Cross-entropy** — hits all models during training/eval with loss. Less critical for pure
   inference serving (vLLM typically doesn't compute loss).

5. **MoE assertions** — hits MoE models (Qwen3-30B-A3B, DeepSeek MoE). The bounds-check
   assertions are potentially eliminable in compiled mode.
