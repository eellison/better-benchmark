# Inductor Improvement PR Stack

Identified improvements to PyTorch Inductor from SOL-gap investigation across
12 models (288 kernels). Ordered by estimated impact (speedup × breadth).

Status: **None submitted yet.** All have local evidence/prototypes in this repo.

---

## PR 1: Graph-level CSE for identical multi-output `where` patterns

**Impact**: 2.93x → ~1x on causal mask kernels (Llama-3.2-1B 16-head attention)  
**Breadth**: Every causal-attention model with >1 head  
**Files to change**: `torch/_inductor/fx_passes/` (new pass or extend existing CSE)  
**Complexity**: Medium  

**Problem**: Inductor writes 16 identical `where(mask, 0, -inf)` outputs to
separate buffers — 60MB of redundant stores. The mask is head-independent but
materialized per-head.

**Fix**: Detect identical `where` nodes with the same condition/true/false
inputs during post-grad graph optimization. Deduplicate to one computation,
return aliases (or a single expand).

**Evidence**: `fixes/fix_causal_mask_cse.py` — benchmarks original vs CSE'd vs
clone variants.

**Repros**:
- `output/aten_repros/dynamo_meta-llama/Llama-3.2-1B_inference/region_007_pointwise_fc69a1a83866_a2fe9c92.py` (Llama causal mask, 16 heads)
- `output/aten_repros/dynamo_BertForMaskedLM_inference/region_013_pointwise_099a98180172_47a5cbda.py` (Bert mask)
- `output/aten_repros/dynamo_OPTForCausalLM_inference/region_*` (OPT causal mask)

---

## ~~PR 2: Eliminate clone realization for innermost-stride-1 tensors~~

**STATUS: ALREADY FIXED in current PyTorch.**

Verified by running the repro with `TORCH_LOGS="output_code"`: the clone is
NOT materialized. Both generated kernels read `arg1_1` directly with stride
`50260*d0 + d1`. No 3.3GB buffer is allocated.

The 2.28x gap on this repro is NOT from clone realization — it's inherent to
the cross-entropy algorithm (2 passes: online_softmax + gather/nll_loss both
read the full 3.2GB input). This is expected behavior.

**Repros** (gap is algorithmic, not a bug):
- `output/aten_repros/dynamo_DistillGPT2_inference/region_007_amax_sum_4cc1548fbf82_da67baef.py`
- `output/aten_repros/dynamo_DistillGPT2_inference/region_008_amax_sum_4cc1548fbf82_da67baef.py`

---

## PR 3: Realize transcendentals when broadcast factor exceeds threshold

**Impact**: 1.35x on RoPE kernels (with CUDA graphs)  
**Breadth**: All RoPE models, any pattern where sin/cos/exp are broadcast  
**Files to change**: `torch/_inductor/ir.py` (`should_realize_on_reuse`), `torch/_inductor/config.py`  
**Complexity**: Low  

**Problem**: cos/sin depend on `[seq, dim]` = 32K unique values but are
broadcast to `[batch, heads, seq, dim]` = 8M elements. Without realization,
Inductor recomputes transcendentals 256x per unique value.

**Fix**: In `StorageBox.should_realize_on_reuse()`, detect expensive ops
(sin/cos/exp/log) and force materialization when expansion factor exceeds
`config.realize_transcendental_recompute_limit` (default=1).

**Evidence**: `fixes/test_rope_sincos.py` — 1.35x with CUDA graphs, slight
regression without (extra kernel launch).

**Repros**:
- `output/aten_repros/dynamo_meta-llama/Llama-3.2-1B_inference/region_008_pointwise_c89a54b74928_0c8c9524.py` (RoPE sin/cos broadcast)
- `output/aten_repros/vllm_Qwen_Qwen3-0.6B_inference/region_000_pointwise_56e37f0cd652_9e00b7fe.py`
- `output/aten_repros/vllm_mistralai_Mistral-7B-Instruct-v0.3_inference/region_011_pointwise_108892bc6e62_8f1de50a.py`

---

## PR 4: GQA Q/K RoPE fusion (3 kernels → 1)

**Impact**: 3 kernel launches → 1 per attention layer  
**Breadth**: All GQA models (Mistral, Qwen3, Llama, gpt-oss-20b)  
**Files to change**: `torch/_inductor/dependencies.py`, `torch/_inductor/lowering.py`, `torch/_inductor/scheduler.py`, `torch/_inductor/config.py`  
**Complexity**: High  

**Problem**: Q RoPE and K RoPE can't fuse because:
- MemoryDep equality ignores broadcast equivalence (cos/sin ranges differ on
  unused dims)
- `expand` forces realization of K RoPE producer via `mark_reuse`
- SIMD backend rejects numel mismatch for horizontal fusion

**Fix** (3 parts):
- A. `MemoryDep.strip_broadcast()` — canonicalize broadcast-equivalent deps
- B. `expand_recompute_limit` config — don't realize when expand ≤ 8x
- C. `fusable_read_and_write` broadcast handling — normalize producer/consumer
  reads through expand patterns

**Evidence**: `output/investigations/02_gqa_qk_rope_fusion.md`

**Repros** (multi-kernel — 3 kernels that should be 1):
- `output/aten_repros/vllm_mistralai_Mistral-7B-Instruct-v0.3_inference/region_009_pointwise_74957135a3e5_6bfb868a.py` (K expand+clone)
- `output/aten_repros/vllm_Qwen_Qwen3-0.6B_inference/region_009_pointwise_74957135a3e5_6bfb868a.py` (same pattern)
- `output/aten_repros/vllm_Qwen_Qwen3-30B-A3B_inference/region_008_pointwise_74957135a3e5_cadcdcc5.py`
- `output/aten_repros/vllm_openai_gpt-oss-20b_inference/region_009_pointwise_962a553f5660_ea80f66a.py`
- `output/aten_repros/dynamo_meta-llama/Llama-3.2-1B_inference/region_004_pointwise_74957135a3e5_7dcf42a6.py`

---

## PR 5: Fix ReductionHint classification for small inner-dim reductions

**Impact**: 2.22x on near-power-of-2 reduction dims (e.g., 513)  
**Breadth**: Any model with reduction dim 65-1024 that incorrectly gets DEFAULT hint  
**Files to change**: `torch/_inductor/scheduler.py` (hint selection), possibly `choices.py`  
**Complexity**: Medium (need to understand hint classification logic)  

**Problem**: `max(dim=-1)` on a 513-wide inner dimension gets
`ReductionHint.DEFAULT` (because `numel >= num_sm*2*32`). DEFAULT's persistent
threshold is only 64, so 513 > 64 → non-persistent loop with R0_BLOCK=512,
needing 2 iterations where the second processes 1/512 useful elements.

**Root question**: Why does this get DEFAULT instead of INNER? It's a
last-dim reduction on a small dimension — should qualify for INNER hint which
has a higher persistent threshold. Blindly bumping DEFAULT's threshold from
64→1024 risks forcing persistence on kernels that legitimately need the
non-persistent path (large reduction dims where register pressure matters).

**Fix direction**: Fix the hint classification to assign INNER for small
inner-dim reductions regardless of numel, rather than changing the DEFAULT
threshold. Alternatively, make the persistent threshold dimension-aware
(persistent when `rdim <= R0_BLOCK` regardless of hint).

**Evidence**: `output/investigations/04_tiling_near_power_of_2.md` —
coordinate descent tuning proves the right config exists (160us → 72us).

**Repros**:
- `output/aten_repros/vllm_openai_gpt-oss-20b_inference/region_019_max_c4bf986c7d26_ac4fc879.py` (max dim=513, sink token)
- `output/aten_repros/vllm_openai_gpt-oss-20b_inference/region_018_max_c3205815ab1d_ac4fc879.py`

**Inductor source**: `/tmp/pytorch-work/torch/_inductor/ir.py:1445-1449` (`Reduction.num_splits()`)

---

## PR 6: Decompose topk to sort+slice for small dimensions

**Impact**: Enables fusion of sigmoid/softmax + topk (eliminates intermediate buffer)  
**Breadth**: All MoE routing (DeepSeek-V3, Qwen3-MoE, gpt-oss-20b)  
**Files to change**: `torch/_inductor/decomposition.py` or lowerings  
**Complexity**: Medium  

**Problem**: `topk` falls back to ATen (separate kernel launch), blocking
fusion with preceding sigmoid/softmax pointwise. For small dims (≤256), sort is
cheap and enables Triton-native fusion.

**Fix**: When `dim_size <= 256`, decompose `topk(x, k)` to
`sort(x, descending=True)[:k]`. Triton handles sort natively, enabling the
entire sigmoid+topk graph to fuse into one kernel.

**Evidence**: `fixes/test_topk_decomp.py`

**Repros**:
- `output/aten_repros/vllm_deepseek-ai_DeepSeek-V3_inference/region_028_pointwise_94263b2bcee2_45691208.py` (sigmoid + bias + topk, dim=32)
- `output/aten_repros/vllm_deepseek-ai_DeepSeek-V3_inference/region_027_sum_76e9103a8afc_21536b54.py` (sum + topk, dim=8)
- `output/aten_repros/vllm_Qwen_Qwen3-30B-A3B_inference/region_015_amax_sum_2db92b37f274_1614f25d.py` (softmax + topk, dim=128)
- `output/aten_repros/vllm_openai_gpt-oss-20b_inference/region_005_pointwise_a9fbc6797886_4a40ec6e.py` (topk, dim=32)

---

## PR 7: Distribute associative reductions over cat inputs

**Impact**: 1.29x on multi-use permute+cat+sum pattern  
**Breadth**: DistillGPT2, any multi-head attention backward with permute+cat+sum  
**Files to change**: `torch/_inductor/fx_passes/post_grad.py`, `torch/_inductor/config.py`  
**Complexity**: Low-Medium  
**Status**: **Branch exists** — `distribute-reduce-over-cat` in `/tmp/pytorch-work`  

**Problem**: Attention backward: 3x `[32,12,512,64]` → permute → reshape →
cat → reshape → sum(dim=0). When the cat output has another consumer (e.g.,
matmul), the 36MB buffer is materialized. The sum then re-reads all 36MB
just to produce a [2304] vector.

**Fix**: General algebraic rewrite in post_grad_passes:
`reduce(cat([x1, x2, ...], dim=D), reduce_dims)` →
`cat([reduce(x1, reduce_dims), reduce(x2, reduce_dims), ...], dim=D')`
when D is orthogonal to reduce_dims. Fires only when the cat has other
users that force materialization (profitability gate).

Not a brittle pattern match — works regardless of what's upstream of
the cat inputs (permute, reshape, pointwise, etc.).

**Evidence**: `fixes/test_outer_sum_permute.py`, benchmarked 1.29x on
multi-use case, neutral on single-use.

**Known limitations**:
- Only handles `aten.sum.dim_IntList` (could extend to prod, amax, amin)
- Only traces through one reshape between cat and sum
- Profitability gate is conservative (multi-user proxy for materialization)

**Repros** (training — DistillGPT2 attention backward):
- `output/aten_repros/dynamo_DistillGPT2/region_004_sum_c15fcea82dde_66ab4406.py`
- `output/aten_repros/dynamo_BertForMaskedLM/region_016_sum_bab7a7b0118a_a53551b1.py` (outer-dim sum after permute)

---

## PR 8: Don't fuse stride-2 producers into much-larger consumers

**Impact**: 2.05x on DeepSeek-V3 interleaved RoPE (221us → 108us)  
**Breadth**: DeepSeek-V3 specific (interleaved RoPE variant)  
**Files to change**: `torch/_inductor/scheduler.py` (fusion scoring)  
**Complexity**: Medium  

**Problem**: Interleave clone (`reshape+permute` with stride-2 innermost) gets
fused into a 3x-larger consumer kernel. The stride-2 pattern runs for all 50M
iterations instead of the 16M where it's needed.

**Fix**: When a producer has non-unit innermost stride and the consumer's
iteration domain is >2x larger (due to cat/expand), score fusion lower or force
realization of the producer. Materializing the interleave separately (22.5us)
then fusing the rest (85us) = 108us vs 221us fused.

**Evidence**: `output/investigations/06_deepseek_interleaved_rope.md`

**Repros**:
- `output/aten_repros/vllm_deepseek-ai_DeepSeek-V3_inference/region_012_pointwise_c0b2590bfdf4_d2f5d45b.py` (interleaved RoPE + Q/K cat, 370MB)

---

## PR 9: Absorb outer-reduction into producing kernel (weight grad sum)

**Impact**: 1.5-2x on weight gradient sums in backward pass  
**Breadth**: All BERT-family training (BertForMaskedLM, etc.)  
**Files to change**: `torch/_inductor/scheduler.py` (multi-output scheduling)  
**Complexity**: High  

**Problem**: `sum([16384, 768], dim=0)` uses split-reduction: kernel 0 writes
workspace, kernel 1 reduces. But the producing kernel already iterates every
element — it could accumulate the sum in registers and atomic_add the result.

**Verified generated code** (3 kernels):
```
K0: triton_poi_fused_clone_mul_permute_view_0
    reads arg0_1[16384,768], writes buf0[32,512,768] (48MB)
K1: triton_red_fused_clone_mul_permute_sum_view_1
    reads buf0[32,512,768] (48MB RE-READ), reduces to buf1[1,768,128] (384KB workspace)
K2: triton_red_fused_clone_mul_permute_sum_view_2
    reads buf1[1,768,128], reduces to buf2[1,768] (final output)
```

K0 writes every element of buf0. K1 re-reads all 48MB just to sum dim=0.
If K0 accumulated a per-column sum while writing (via per-CTA accumulators +
atomic_add), it would eliminate: the 48MB re-read + 384KB workspace + K1 + K2.

**Fix**: Detect when a split-reduction's input is the sole output of a
preceding pointwise kernel that iterates the same domain. Absorb the reduction
as an accumulator + atomic_add in the producer. Hard because it requires
cross-CTA coordination that the current scheduler doesn't model.

**Evidence**: `fixes/fix_weight_grad_sum.py` — single-kernel Triton with
atomic_add, verified correct, ~1.2x speedup.

**Repros** (training — BertForMaskedLM backward):
- `output/aten_repros/dynamo_BertForMaskedLM/region_017_sum_68c1d76d5051_ea349839.py` (sum [16384,768] dim=0)
- `output/aten_repros/dynamo_BertForMaskedLM/region_018_sum_36490c41a058_a53551b1.py`
- `output/aten_repros/dynamo_BertForMaskedLM/region_019_sum_61a7becdb42d_33c89a90.py`
- `output/aten_repros/dynamo_BertForMaskedLM/region_020_sum_e90d302edc1d_bcd9ce40.py`

---

## PR 10: ~~Use `tl.sigmoid` instead of `1/(1+exp(-x))` decomposition~~

**STATUS: INVALID — numerically incorrect.**

`tl.sigmoid` uses SFU hardware approximation (~2^-22 relative error) which is
NOT equivalent to the decomposed full-precision f32 path. This would break
training (gradient accumulation amplifies the error) and fail bitwise/tight-
tolerance correctness tests.

Would require a precision-aware gate (`config.use_fast_math_sigmoid`) disabled
by default, plus evidence that the error is below noise floor for specific
inference-only use cases. Not worth pursuing as a general Inductor change.

**Evidence**: `output/investigations/08_misc_patterns.md` (section 8a) — the
1.68x gap is real but the proposed fix is wrong.

---

## Suggested Landing Order

**Tier 1 — Low risk, high confidence** (land first):
1. PR 3: Transcendental realization (new config knob, easy to disable)
2. PR 6: topk decomposition (guarded by dim-size check)
3. PR 1: CSE pass (new pass, additive)

**Tier 2 — Medium scope, strong evidence**:
4. PR 5: Reduction hint fix (needs investigation into why hint is wrong)
5. PR 7: Distribute reduce over cat (**branch: `distribute-reduce-over-cat`**)
6. PR 8: Fusion scoring for stride-2 producers

**Tier 3 — Larger changes, need careful testing**:
7. PR 4: GQA fusion (touches scheduler core)
8. PR 9: Producer-absorbed reduction (multi-output scheduling)

**Dropped**:
- ~~PR 2~~: clone realization — already fixed in current PyTorch
- ~~PR 10~~: sigmoid — numerically incorrect, not a valid optimization

---

## Not Included (separate stack)

These are **repro infrastructure / benchmarking improvements**, not Inductor changes:
- Path whitespace normalization in `bench.py`
- Loader shims for `device`, `inf`, `nan`, `randn(dtype=bool)`
- GPU advisory lock improvements
- Repro-quality issue board
