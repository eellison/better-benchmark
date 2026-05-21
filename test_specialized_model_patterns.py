"""
Test: Model the patterns from vllm PR #38595 specialized DeepSeek V3.2 NVFP4
as pure aten ops, check what inductor fuses.

The PR has two hand-fused Triton kernels:
  1. _fused_norm_rope_kernel (4 PIDs per token):
     pid0: LayerNorm(index_k) → RoPE(neox) → FP8 quant(ue8m0) → cache_write
     pid1: RMSNorm(kv_c) + RoPE(k_pe, interleaved) → concat → cache_write
     pid2: RMSNorm(q_c) → output
     pid3: fill topk_indices with -1

  2. _fused_q_kernel (3 PIDs per token per head):
     pid0: RoPE(q_pe, interleaved) → FP8 quant(static) → pack into mqa_q tail
     pid1: RoPE(index_q, neox) → FP8 quant(ue8m0) → scale_weights
     pid2: FP8 quant(ql_nope, static) → pack into mqa_q front

Question: which of these would inductor fuse automatically if decomposed?
What WOULDN'T fuse (requiring hand-written kernels)?
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
from torch._inductor.graph import GraphLowering

cfg.force_disable_caches = True

original_update = GraphLowering._update_scheduler


def fusion_reporter(self):
    from torch._inductor.scheduler import Scheduler
    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)
    nodes = self.scheduler.nodes
    fused = [nd for nd in nodes if hasattr(nd, 'snodes') and len(nd.snodes) > 1]
    single = len(nodes) - len(fused)
    print(f"    Scheduler: {len(nodes)} nodes ({len(fused)} fused, {single} single)")
    for nd in nodes:
        if hasattr(nd, 'snodes') and len(nd.snodes) > 1:
            names = [s.node.get_name() for s in nd.snodes if hasattr(s, 'node')]
            print(f"      Fused({len(nd.snodes)}): {names[:10]}{'...' if len(names) > 10 else ''}")
        else:
            name = nd.node.get_name() if hasattr(nd, 'node') else '?'
            ntype = type(nd).__name__
            print(f"      {ntype}: {name}")


# ============ Helpers ============

def rms_norm(x, weight, eps=1e-6):
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    return (x_f32 * torch.rsqrt(variance + eps)).to(x.dtype) * weight


def layer_norm(x, weight, bias, eps=1e-6):
    x_f32 = x.float()
    mean = x_f32.mean(dim=-1, keepdim=True)
    var = x_f32.var(dim=-1, keepdim=True, correction=0)
    normed = (x_f32 - mean) * torch.rsqrt(var + eps)
    return (normed * weight.float() + bias.float()).to(x.dtype)


def rope_neox(x, cos, sin):
    """RoPE neox-style: split halves, rotate."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def rope_interleaved(x, cos, sin):
    """RoPE interleaved: pairs at (2i, 2i+1)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    r1 = x1 * cos - x2 * sin
    r2 = x2 * cos + x1 * sin
    return torch.stack([r1, r2], dim=-1).flatten(-2)


def fp8_quant_static(x, scale):
    """FP8 quantize with static scale."""
    return (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def fp8_quant_ue8m0(x):
    """FP8 quantize with per-row power-of-2 scale (ue8m0)."""
    x_f32 = x.float()
    amax = x_f32.abs().amax(dim=-1, keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2(amax.clamp(min=1e-4) / 448.0)))
    x_fp8 = (x_f32 / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return x_fp8, scale.squeeze(-1)


# ============================================================
# Pattern 1a (pid0): LayerNorm → RoPE(neox) → FP8 quant
# This is the index-K path
# ============================================================
print("=" * 70)
print("Pattern 1a (pid0): LayerNorm → RoPE(neox) → FP8 quant(ue8m0)")
print("  index_k path: norm → rotate → quantize")
print("=" * 70)


def pattern_1a_index_k(index_k, ln_w, ln_b, cos, sin):
    # LayerNorm
    normed = layer_norm(index_k, ln_w, ln_b)
    # RoPE neox (only first 2*half_rot_dim elements get rotated)
    roped = rope_neox(normed, cos, sin)
    # FP8 quantize (ue8m0 per-row)
    fp8_out, scale = fp8_quant_ue8m0(roped)
    return fp8_out, scale


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
tokens = 4
INDEX_K_DIM = 256
HALF_ROT = 64  # half of rot dim
index_k = torch.randn(tokens, INDEX_K_DIM, device="cuda", dtype=torch.bfloat16)
ln_w = torch.randn(INDEX_K_DIM, device="cuda", dtype=torch.bfloat16)
ln_b = torch.randn(INDEX_K_DIM, device="cuda", dtype=torch.bfloat16)
cos_1a = torch.randn(tokens, 1, device="cuda", dtype=torch.bfloat16).expand(tokens, INDEX_K_DIM // 2)
sin_1a = torch.randn(tokens, 1, device="cuda", dtype=torch.bfloat16).expand(tokens, INDEX_K_DIM // 2)
compiled_1a = torch.compile(pattern_1a_index_k, dynamic=True)
with torch.no_grad():
    compiled_1a(index_k, ln_w, ln_b, cos_1a, sin_1a)


# ============================================================
# Pattern 1b (pid1): RMSNorm(kv_c) + RoPE(k_pe, interleaved) → concat
# This is the MLA KV cache write path
# ============================================================
print("\n" + "=" * 70)
print("Pattern 1b (pid1): RMSNorm(kv_c) + RoPE(k_pe, interleaved) → concat")
print("  MLA kv path: normalize kv_c, rotate k_pe, concat for cache")
print("=" * 70)


def pattern_1b_mla_kv(kv_c, kv_norm_w, k_pe, cos, sin):
    # RMSNorm on kv_c
    kv_normed = rms_norm(kv_c, kv_norm_w)
    # RoPE interleaved on k_pe
    k_pe_roped = rope_interleaved(k_pe.float(), cos, sin)
    # Concat for MLA cache: [kv_normed | k_pe_roped]
    cache_entry = torch.cat([kv_normed, k_pe_roped.to(kv_normed.dtype)], dim=-1)
    return cache_entry


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
KV_DIM = 512
KPE_DIM = 64  # interleaved pairs
kv_c = torch.randn(tokens, KV_DIM, device="cuda", dtype=torch.bfloat16)
kv_norm_w = torch.randn(KV_DIM, device="cuda", dtype=torch.bfloat16)
k_pe = torch.randn(tokens, KPE_DIM, device="cuda", dtype=torch.bfloat16)
cos_1b = torch.randn(tokens, KPE_DIM // 2, device="cuda", dtype=torch.float32)
sin_1b = torch.randn(tokens, KPE_DIM // 2, device="cuda", dtype=torch.float32)
compiled_1b = torch.compile(pattern_1b_mla_kv, dynamic=True)
with torch.no_grad():
    compiled_1b(kv_c, kv_norm_w, k_pe, cos_1b, sin_1b)


# ============================================================
# Pattern 1b+FP8: Same but with FP8 cache (MLA_CACHE_FP8=True)
# RMSNorm(kv_c) + RoPE(k_pe) → concat → FP8 quant(static) → cache
# ============================================================
print("\n" + "=" * 70)
print("Pattern 1b+FP8: RMSNorm + RoPE → concat → FP8 quant (static scale)")
print("  MLA kv with FP8 cache: norm + rope + concat + quant all fused?")
print("=" * 70)


def pattern_1b_fp8(kv_c, kv_norm_w, k_pe, cos, sin, mla_k_scale):
    kv_normed = rms_norm(kv_c, kv_norm_w)
    k_pe_roped = rope_interleaved(k_pe.float(), cos, sin)
    cache_entry = torch.cat([kv_normed, k_pe_roped.to(kv_normed.dtype)], dim=-1)
    # FP8 quant with static scale for cache
    cache_fp8 = fp8_quant_static(cache_entry, mla_k_scale)
    return cache_fp8


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
mla_k_scale = torch.tensor(0.05, device="cuda", dtype=torch.float32)
compiled_1b_fp8 = torch.compile(pattern_1b_fp8, dynamic=True)
with torch.no_grad():
    compiled_1b_fp8(kv_c, kv_norm_w, k_pe, cos_1b, sin_1b, mla_k_scale)


# ============================================================
# Pattern 1c (pid2): RMSNorm(q_c) → output
# Simple standalone norm
# ============================================================
print("\n" + "=" * 70)
print("Pattern 1c (pid2): RMSNorm(q_c) alone")
print("  Just a norm — trivially 1 kernel")
print("=" * 70)


def pattern_1c_q_norm(q_c, q_norm_w):
    return rms_norm(q_c, q_norm_w)


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
Q_DIM = 1536
q_c = torch.randn(tokens, Q_DIM, device="cuda", dtype=torch.bfloat16)
q_norm_w = torch.randn(Q_DIM, device="cuda", dtype=torch.bfloat16)
compiled_1c = torch.compile(pattern_1c_q_norm, dynamic=True)
with torch.no_grad():
    compiled_1c(q_c, q_norm_w)


# ============================================================
# Pattern 1-ALL: All 4 pid programs as one function
# Can inductor fuse across independent data-parallel chains?
# ============================================================
print("\n" + "=" * 70)
print("Pattern 1-ALL: All 4 pids combined (the full fused_norm_rope kernel)")
print("  pid0: LayerNorm→RoPE→FP8, pid1: RMSNorm+RoPE→cat,")
print("  pid2: RMSNorm(q_c), pid3: fill(-1)")
print("  Question: does inductor emit 1 kernel or 4?")
print("=" * 70)


def pattern_1_all(
    index_k, ln_w, ln_b, cos_idx, sin_idx,
    kv_c, kv_norm_w, k_pe, cos_kpe, sin_kpe, mla_k_scale,
    q_c, q_norm_w,
    topk_buffer,
):
    # pid0: index_k path
    idx_normed = layer_norm(index_k, ln_w, ln_b)
    idx_roped = rope_neox(idx_normed, cos_idx, sin_idx)
    idx_fp8, idx_scale = fp8_quant_ue8m0(idx_roped)

    # pid1: MLA kv path
    kv_normed = rms_norm(kv_c, kv_norm_w)
    k_pe_roped = rope_interleaved(k_pe.float(), cos_kpe, sin_kpe)
    mla_entry = torch.cat([kv_normed, k_pe_roped.to(kv_normed.dtype)], dim=-1)
    mla_fp8 = fp8_quant_static(mla_entry, mla_k_scale)

    # pid2: q_c norm
    q_normed = rms_norm(q_c, q_norm_w)

    # pid3: fill topk buffer
    topk_out = torch.full_like(topk_buffer, -1)

    return idx_fp8, idx_scale, mla_fp8, q_normed, topk_out


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
TOPK = 256
topk_buf = torch.zeros(tokens, TOPK, device="cuda", dtype=torch.int32)
compiled_all1 = torch.compile(pattern_1_all, dynamic=True)
with torch.no_grad():
    compiled_all1(
        index_k, ln_w, ln_b, cos_1a, sin_1a,
        kv_c, kv_norm_w, k_pe, cos_1b, sin_1b, mla_k_scale,
        q_c, q_norm_w,
        topk_buf,
    )


# ============================================================
# Pattern 2a (pid0): RoPE(interleaved) → FP8 quant(static) → pack
# q_pe path
# ============================================================
print("\n" + "=" * 70)
print("Pattern 2a (pid0): RoPE(q_pe, interleaved) → FP8 quant(static)")
print("  q_pe rotation + quantize into mqa_q buffer")
print("=" * 70)


def pattern_2a_qpe(q_pe, cos, sin, q_scale):
    # RoPE interleaved
    q_pe_roped = rope_interleaved(q_pe.float(), cos, sin)
    # FP8 quant static
    q_pe_fp8 = fp8_quant_static(q_pe_roped, q_scale)
    return q_pe_fp8


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
NUM_HEADS = 64
QPE_DIM = 64  # interleaved
q_pe_in = torch.randn(tokens, NUM_HEADS, QPE_DIM, device="cuda", dtype=torch.bfloat16)
cos_2a = torch.randn(tokens, 1, QPE_DIM // 2, device="cuda", dtype=torch.float32)
sin_2a = torch.randn(tokens, 1, QPE_DIM // 2, device="cuda", dtype=torch.float32)
q_scale = torch.tensor(0.02, device="cuda", dtype=torch.float32)
compiled_2a = torch.compile(pattern_2a_qpe, dynamic=True)
with torch.no_grad():
    compiled_2a(q_pe_in, cos_2a, sin_2a, q_scale)


# ============================================================
# Pattern 2b (pid1): RoPE(neox) → FP8 quant(ue8m0) → scale weights
# index_q path
# ============================================================
print("\n" + "=" * 70)
print("Pattern 2b (pid1): RoPE(index_q, neox) → FP8 quant(ue8m0) + scale_weights")
print("  index_q rotation + dynamic quant + weight scaling")
print("=" * 70)


def pattern_2b_index_q(index_q, cos, sin, index_weights, softmax_scale, head_scale):
    # RoPE neox on index_q
    roped = rope_neox(index_q, cos, sin)
    # FP8 quant with per-head ue8m0 scale
    fp8_out, quant_scale = fp8_quant_ue8m0(roped)
    # Scale the weights: w *= quant_scale * softmax_scale * head_scale
    scaled_weights = index_weights * quant_scale * softmax_scale * head_scale
    return fp8_out, scaled_weights


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
NUM_INDEX_HEADS = 32
INDEX_Q_DIM = 192
index_q_in = torch.randn(tokens, NUM_INDEX_HEADS, INDEX_Q_DIM, device="cuda", dtype=torch.bfloat16)
cos_2b = torch.randn(tokens, 1, INDEX_Q_DIM // 2, device="cuda", dtype=torch.bfloat16)
sin_2b = torch.randn(tokens, 1, INDEX_Q_DIM // 2, device="cuda", dtype=torch.bfloat16)
index_weights_in = torch.randn(tokens, NUM_INDEX_HEADS, device="cuda", dtype=torch.float32)
softmax_sc = torch.tensor(0.125, device="cuda", dtype=torch.float32)
head_sc = torch.tensor(1.0, device="cuda", dtype=torch.float32)
compiled_2b = torch.compile(pattern_2b_index_q, dynamic=True)
with torch.no_grad():
    compiled_2b(index_q_in, cos_2b, sin_2b, index_weights_in, softmax_sc, head_sc)


# ============================================================
# Pattern 2c (pid2): FP8 quant(ql_nope, static) — trivial
# ============================================================
print("\n" + "=" * 70)
print("Pattern 2c (pid2): FP8 quant(ql_nope, static scale)")
print("  Simple static quantize — trivially 1 kernel")
print("=" * 70)


def pattern_2c_nope_quant(ql_nope, q_scale):
    return fp8_quant_static(ql_nope, q_scale)


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
QL_NOPE_DIM = 512
ql_nope_in = torch.randn(tokens, NUM_HEADS, QL_NOPE_DIM, device="cuda", dtype=torch.bfloat16)
compiled_2c = torch.compile(pattern_2c_nope_quant, dynamic=True)
with torch.no_grad():
    compiled_2c(ql_nope_in, q_scale)


# ============================================================
# Pattern 2-ALL: Full fused_q as one function
# RoPE+quant on q_pe, RoPE+quant on index_q, quant on ql_nope
# Then PACK: mqa_q = cat([ql_nope_fp8, q_pe_fp8], dim=-1)
# ============================================================
print("\n" + "=" * 70)
print("Pattern 2-ALL: Full fused_q kernel as one function")
print("  pid0: q_pe rope+quant, pid1: index_q rope+quant+scale,")
print("  pid2: ql_nope quant, then pack mqa_q = cat([nope, pe])")
print("  Question: does inductor emit 1 kernel or multiple?")
print("=" * 70)


def pattern_2_all(
    q_pe, cos_qpe, sin_qpe, q_scale,
    index_q, cos_idx, sin_idx, index_weights, softmax_scale, head_scale,
    ql_nope,
):
    # pid0: q_pe RoPE + FP8 quant
    q_pe_roped = rope_interleaved(q_pe.float(), cos_qpe, sin_qpe)
    q_pe_fp8 = fp8_quant_static(q_pe_roped, q_scale)

    # pid1: index_q RoPE + FP8 quant + weight scale
    idx_roped = rope_neox(index_q, cos_idx, sin_idx)
    idx_fp8, idx_qscale = fp8_quant_ue8m0(idx_roped)
    scaled_weights = index_weights * idx_qscale * softmax_scale * head_scale

    # pid2: ql_nope FP8 quant
    nope_fp8 = fp8_quant_static(ql_nope, q_scale)

    # Pack mqa_q: [nope_fp8 | pe_fp8] along last dim
    mqa_q = torch.cat([nope_fp8, q_pe_fp8], dim=-1)

    return mqa_q, idx_fp8, scaled_weights


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
compiled_all2 = torch.compile(pattern_2_all, dynamic=True)
with torch.no_grad():
    compiled_all2(
        q_pe_in, cos_2a, sin_2a, q_scale,
        index_q_in, cos_2b, sin_2b, index_weights_in, softmax_sc, head_sc,
        ql_nope_in,
    )


# ============================================================
# Pattern CACHE WRITE: norm → quant → scatter (paged cache write)
# The KEY pattern the hand-kernel fuses that inductor might not
# ============================================================
print("\n" + "=" * 70)
print("Pattern CACHE: norm → quant → scatter_nd (paged cache write)")
print("  This is what makes the hand-kernel unique: fusing the WRITE")
print("=" * 70)


def pattern_cache_write(
    index_k, ln_w, ln_b, cos, sin,
    slot_mapping,   # [tokens] -> slot indices
    kv_cache,       # [num_blocks, block_size, head_dim] fp8
):
    # Compute the values
    normed = layer_norm(index_k, ln_w, ln_b)
    roped = rope_neox(normed, cos, sin)
    fp8_out, scale = fp8_quant_ue8m0(roped)

    # Paged cache write (scatter)
    block_size = kv_cache.shape[1]
    block_idx = slot_mapping // block_size
    block_offset = slot_mapping % block_size
    # scatter into cache
    kv_cache[block_idx, block_offset, :] = fp8_out

    return kv_cache, scale


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
NUM_BLOCKS = 16
BLOCK_SIZE = 16
slot_mapping = torch.randint(0, NUM_BLOCKS * BLOCK_SIZE, (tokens,), device="cuda")
kv_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, INDEX_K_DIM, device="cuda", dtype=torch.float8_e4m3fn)
compiled_cache = torch.compile(pattern_cache_write, dynamic=True)
with torch.no_grad():
    compiled_cache(index_k, ln_w, ln_b, cos_1a, sin_1a, slot_mapping, kv_cache)


GraphLowering._update_scheduler = original_update
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
What inductor CAN fuse (matching the hand-kernel):
  - LayerNorm → RoPE → FP8 quant: YES, all pointwise, 1 kernel
  - RMSNorm → RoPE → concat: YES, pointwise + cat, 1 kernel
  - RMSNorm → RoPE → concat → FP8: YES, all 1 kernel
  - RoPE → FP8 quant (static): YES, trivially pointwise
  - RoPE → FP8 quant (ue8m0) → scale: YES, 1 kernel
  - cat([nope_fp8, pe_fp8]): YES, pointwise_cat into consumer

What inductor CANNOT fuse (requiring hand-written kernels):
  1. MULTI-PID PARALLELISM: The hand-kernel runs 4 independent programs
     (pid0-3) in a SINGLE kernel launch. Inductor emits them as separate
     kernels — same total FLOPs but more launches & scheduling overhead.

  2. CACHE WRITE FUSION: scatter_nd into paged KV cache fused with the
     compute (norm+rope+quant). Inductor may or may not fuse the scatter
     with the preceding pointwise chain.

  3. CROSS-HEAD PARALLELISM: fused_q runs 3 pids per (token, head),
     packing results from different heads into the same output tensor.
     Inductor would generate separate kernels per logical operation.

Bottom line: The hand-kernels don't find fusions inductor can't — they
reduce KERNEL LAUNCH COUNT by packing independent work into one dispatch.
This matters at low token counts (decode) where launch overhead dominates.
""")
