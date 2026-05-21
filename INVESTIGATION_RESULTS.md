# Fusion Region SOL Gap Investigation Results

GPU: NVIDIA B200 (SM100), 148 SMs, CUDA 13.0, PyTorch 2.12.0a0
Benchmarked: 12 models, 288 kernels, 42 moderate-gap targets investigated

## Overall Classification (288 kernels)

| Category | Count | % | Description |
|---|---|---|---|
| Compute-bound (<0.8× SOL) | 123 | 42% | Faster than memcopy |
| Near SOL (0.8-1.3×) | 65 | 22% | Well-optimized |
| **Moderate gap (1.3-3×)** | **42** | **14%** | **Optimization targets** |
| Launch-bound (>3×) | 58 | 20% | Kernel launch dominates |

## Root Cause Analysis by Category

### 1. Cross-Entropy Loss (amax_sum) — REVISED: Mostly at SOL already

**Pattern**: `logsumexp(logits, vocab_dim)` + masked gather + sum (from `ForCausalLMLoss`)

**UPDATED finding** (from deeper investigation with fresh compilation):
- **Albert [4096, 30000]**: 1.03× gap — effectively AT SOL
- **OPT [8192, 50272]**: 0.98× — AT SOL (faster than memcopy!)
- **DistillGPT2 [16384, 50257]**: 1.60× — real gap, caused by NON-CONTIGUOUS STRIDES

Inductor's online softmax algorithm is excellent — it achieves near-peak bandwidth (3137-3396 GB/s).

**Root cause for DistillGPT2**: Input has stride=[50260,1] (not contiguous). Inductor fuses a
`clone(contiguous_format)` INTO the reduction kernel, doubling memory traffic (read strided + write contiguous).

**Fix**: Eliminate the clone by having the gather kernel handle strided access directly.
Not a general Inductor issue — only fires with non-contiguous inputs from transformers' `.contiguous()` call.

---

### 2. RMSNorm (mean) — 1.4-2.6× gap, 38-40us

**Pattern**: `bf16→f32 → pow(2) → mean(dim=-1) → rsqrt → mul → bf16 → weight_mul`

**Root cause varies by hidden dimension**:
- **hidden_dim=2048 (Llama-3.2-1B)**: Persistent kernel (R0_BLOCK=2048), single-pass, but LOW OCCUPANCY. 2048 f32 values per thread block = 8KB registers. Few waves fit on SM → poor latency hiding → 2.6× gap.
- **hidden_dim=7168 (DeepSeek-V3)**: Multi-pass `triton_red` kernel. Reads input TWICE (first loop: accumulate sum-of-squares; second loop: normalize and write). Double traffic → 1.77× gap.

**Fix**:
- For dim≤4096: Increase XBLOCK to improve occupancy (trade parallelism for waves)
- For dim>4096: Use larger R0_BLOCK on SM100 (64K regs/SM), or Welford single-pass approach
- vLLM's fused RMS+quant CUDA kernel avoids this entirely

---

### 3. LayerNorm + GELU (var_mean) — 2.4-2.7× gap, 42-71us

**Pattern**: `reshape → GELU(mul,erf,add) → var_mean(dim=2, correction=0)`

**Root cause**: Register pressure from fused computation.
- Persistent kernel with R0_BLOCK=1024 for r0_numel=768
- **4 Welford accumulators** (sum, sum_sq, count, correction) × R0_BLOCK = massive register usage
- GELU's `erf` adds ~5 FP ops inline
- Combined register pressure → low occupancy → poor latency hiding

**Fix**:
- Reduce to 2-accumulator Welford (mean, M2) — saves 50% register pressure
- Or: separate GELU from var_mean (2 kernels but each has better occupancy)
- Note: BERT's LayerNorm uses `var_mean` (needs both outputs) vs Llama's simpler `mean`

---

### 4. Weight Gradient Sum — 1.5-2.0× gap, 53-97us

**Pattern**: `sum([16384, 768], dim=0) → [768]` (outer-dim reduction in backward pass)

**Root cause**: Split-reduction with workspace materialization.
- 16384 rows exceed cooperative thread limit for single-kernel reduction
- **Kernel 0**: Each CTA processes RSPLIT_SIZE rows → accumulates to workspace (384KB)
- **Kernel 1**: Reduces workspace partials → final [768] output
- Critical: The producing kernel already writes every element of the buffer that gets summed, but **doesn't absorb the reduction** into its write loop

**Fix**: Absorb outer-reduction into the producing kernel's write loop as an accumulator. The producing kernel already touches every element — adding `accum += value` per iteration costs nothing vs the GMEM re-read. This is a multi-output scheduling limitation in Inductor.

---

### 5. RoPE / Attention Pointwise — 2.2-2.9× gap, 35-68us

**Multiple sub-patterns with different root causes**:

#### Region 007 (Causal Mask, 2.93×): **16 IDENTICAL OUTPUTS**
- Computes one causal mask, writes it to **16 separate buffers** (all bit-identical)
- Graph-level CSE failure — should be deduplicated to 1 output
- Fix: Deduplicate at graph level → should drop to ~1×

#### Region 005 (RoPE, 2.56×): **Scattered loads + masked accesses**
- 10 loads, 2 stores. Q read from 2 offsets for rotate_half
- `tl.load(..., mask=x0<32)` wastes half the threads per load
- KV 4× repeat expansion: same computation done 4× with different write offsets
- Fix: Split into Q-only and KV-only kernels; eliminate masked loads

#### Region 008 (Transpose, 1.53×): **Classic 1D transpose problem**
- Non-contiguous read pattern from `[4,64,512,64]→[4,512,64,64]` permute
- Inductor's 1D pointwise kernel can't do 2D-tiled transpose with shared memory
- 1.53× is near-optimal for naive 1D approach
- Fix: 2D-tiled transpose with shared memory staging

#### Region 001 (LayerNorm affine, 2.46×): **Instruction density**
- 5 loads + 5 FP ops + 1 store per element = 11 instructions
- Reduces memory-level parallelism: only ~5.8 loads/cycle/SM vs memcopy's ~32
- Fix: Not easily fixable without reducing fused operations

---

### 6. MoE / Small Pointwise — 1.3-2.1× gap, 29-51us

**Pattern**: Small tensor operations (14-50KB) from MoE routing, weight permutes

**Root cause**: Kernel launch overhead is 40-60% of total time.
- Kernel launch + grid scheduling: ~10-15us minimum on B200
- Actual computation: ~15-25us
- Combined: inherently 1.3-2× vs theoretical memcopy SOL
- Some have permute/transpose adding non-sequential access

**Fix**: MOSTLY UNFIXABLE at these tensor sizes. Only horizontal fusion (batching multiple small ops into one kernel launch) could help. This is what graph-level operation batching addresses.

---

## Summary: Actionable Optimizations

### High Impact (fixes in Inductor)
1. **CSE deduplication** for identical multi-output patterns (region 007: 2.93× → ~1×)
2. **Absorb outer-reduction into producer** for weight gradient sums (1.5-2× → ~1.1×)
3. **2-accumulator Welford** for LayerNorm var_mean (2.7× → ~1.5×)
4. **Don't decompose cross-entropy** (already correct in vLLM)

### Medium Impact (Inductor tuning)
5. **Occupancy tuning** for RMSNorm persistent kernels (XBLOCK search)
6. **2D-tiled transpose** for attention reshapes (1.53× → ~1.1×)
7. **Split rotate_half** to eliminate masked loads (2.56× → ~1.5×)

### Low/No Impact (inherent limitations)
8. MoE/small pointwise: launch overhead, needs horizontal fusion
9. Instruction-dense pointwise: memory-level parallelism limit
