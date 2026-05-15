# Single-Kernel SOL Gap Analysis

## Summary

Benchmarked 375 single-kernel compilation regions across 33 models on B200 GPUs.
Measured default compile, coordinate descent tuning, and CUDA graph replay.

- **298 real kernels** (k>=1, non-zero bytes) — 55 were no-ops (views/reshapes), 22 errors
- Default compile: **median 5.25x SOL, mean 4.77x**
- CUDA graph: **median 0.96x SOL, mean 1.08x** — nearly all kernels reach SOL when launch overhead is removed
- Only **17 kernels remain >= 2x SOL with CUDA graph** — these have actual kernel quality issues

**The dominant issue is kernel launch latency, not kernel code quality.**

## Methodology

For each repro:
1. Compile with `torch.compile` (default) and with `coordinate_descent_tuning=True`
2. Capture into `torch.cuda.CUDAGraph` and measure graph replay time
3. Measure timing: 10 warmup + 100 iterations for each variant
4. Measure memcopy SOL at same byte count using `dst.copy_(src)` (read+write = `total_bytes`)
5. Byte counting: `torch._inductor.metrics.num_bytes_accessed` (inductor's own count, accounts for views, expands, etc.)

## CUDA Graph Proves It's Launch Overhead

| Metric | Default Compile | CUDA Graph |
|---|---|---|
| Median SOL gap | 5.25x | 0.96x |
| Mean SOL gap | 4.77x | 1.08x |
| Kernels >= 2x SOL | 227 | 17 |
| Kernels >= 5x SOL | 160 | 2 |

CUDA graph replay eliminates per-kernel launch overhead by recording the kernel sequence
and replaying it as a single GPU submission. The median kernel goes from 5.25x SOL to 0.96x SOL —
meaning the **actual triton/CUDA kernel code is nearly optimal** for bandwidth utilization.

Graph speedup distribution (298 kernels):
- **119 kernels (40%)**: >5x speedup from graph — pure launch overhead
- **115 kernels (39%)**: 2-5x speedup — mostly launch overhead
- **18 kernels (6%)**: 1.2-2x speedup — some overhead + kernel issues
- **46 kernels (15%)**: <1.2x speedup — already large enough to amortize overhead

## Kernels Still Slow With CUDA Graph (17 kernels)

These are the only kernels with actual code quality issues — still >= 2x SOL even in graph mode.

| Graph Gap | Pattern | Data | Root Cause |
|---|---|---|---|
| **5.81x** | Reformer index_select (8 batched) | 64 MB | Random memory access from indirect indexing |
| **5.44x** | MoE softmax + topk (Qwen3-MoE) | 2.7 MB | topk is inherently hard to parallelize |
| **4.79x** | MoE sigmoid + topk (DeepSeek-V3) | 6.3 MB | Same topk issue |
| **3.73x** | DeepSeek-V3 RoPE interleave + Q/K | 352 MB | Strided permute [0,1,2,4,3] + clone |
| **3.48x** | GPT-OSS attn softmax + causal mask + max | 256 MB | Softmax reduction on [4,64,512,513] |
| **3.21x** | GPT-OSS attn softmax variant | 256 MB | Same pattern, different layer |
| **3.13x** | GELU + var_mean (Albert) | 48 MB | Short reduction dim after compute |
| **3.06x** | GPT-OSS attn softmax + cat(sinks) | 260 MB | cat + softmax on [512,513] |
| **2.6-2.8x** | var_mean / GELU+var_mean | 24-48 MB | Reduction dim 768-1024, awkward for tiling |
| **2.46x** | causal mask + any reduction (OPT) | 64 MB | Fused mask construction + any() |
| **2.2x** | MT5 relative pos bias + softmax+mask | 24 MB | embedding + softmax + any fused |
| **2.08x** | GQA repeat_kv (Mistral) | 20 MB | expand + clone for head replication |

### Fixable patterns:

1. **var_mean / RMSNorm reductions** (5 kernels, 2.6-3.1x): Could benefit from persistent reductions.
   Reduction dims of 768-1024 are in the awkward range — too large for warp-level, 
   suboptimal for current tiling.

2. **Strided permute + clone** (1 kernel, 3.73x): The interleaved RoPE pattern does
   `reshape([b,h,s,d//2,2]) → permute([0,1,2,4,3]) → clone`. A specialized transpose
   kernel using shared memory tiling would help.

3. **TopK** (2 kernels, 4.8-5.4x): Inherently hard. Could potentially use approximate
   topk or specialized radix-select for the common case of k=4 or k=8.

4. **GPT-OSS softmax on [512,513]** (3 kernels, 3.0-3.5x): The +1 from attention sinks
   creates a non-power-of-2 inner dimension that may hurt tiling.

## Kernel Categories (118 unique patterns, >=2x SOL)

### Latency-Bound Categories (130 kernels)

These all hit the 35-40µs floor. The "gap" is dominated by launch overhead, not kernel inefficiency.

| Category | N | Worst | Median | Key Pattern |
|---|---|---|---|---|
| TRANSPOSE_COPY | 10 | 10.1x | 6.2x | permute + clone for attention reshape |
| PERMUTE_RESHAPE | 6 | 8.6x | 7.8x | permute for matmul prep (no copy) |
| RMSNORM | 15 | 9.4x | 5.5x | pow + mean + rsqrt + mul (LLM layers) |
| VAR_MEAN | 17 | 10.2x | 5.0x | HF LayerNorm via var_mean |
| LAYERNORM_MANUAL | 18 | 7.7x | 2.7x | sub + rsqrt + mul + add pattern |
| SILU_ACTIVATION | 5 | 7.8x | 7.1x | neg + exp + add + div + mul |
| TRIVIAL_FILL/SCALAR | 10 | 7.8x | 5.9x | full, select, scalar ops |
| TINY_POINTWISE | 6 | 8.6x | 6.9x | ne, lt on small tensors |
| GQA_REPEAT_KV | 3 | 7.3x | 7.1x | unsqueeze + expand + clone for GQA |
| ROPE_EMBED | 2 | 9.3x | 8.7x | cos/sin computation for RoPE |
| PAD_CAT | 2 | 7.8x | 7.7x | full + cat for padding |
| MASK_CONSTRUCT | 1 | 7.0x | 7.0x | attention mask construction |
| COMPARISON | 1 | 6.5x | 6.5x | ne/lt for position IDs |
| RESIDUAL_ADD | 1 | 6.2x | 6.2x | single add (residual connection) |
| RELU | 1 | 5.9x | 5.9x | single relu |

### Bandwidth-Bound Categories (18 kernels)

These genuinely take >45µs and are slow relative to data moved.

| Category | N | Worst | Key Issue |
|---|---|---|---|
| TOPK/MOE_ROUTING | 3 | 13.4x | topk is inherently hard to parallelize; sort/select pattern |
| ATTN_MASK_INDEX | 3 | 11.6x | indirect indexing via `aten.index.Tensor` destroys coalescing |
| EMBEDDING+SOFTMAX | 2 | 5.7x | embedding table + relative position bias + softmax |
| ATTN_MASK+ANY | 1 | 3.7x | causal mask + any reduction for unmasking |
| Large LayerNorm | ~5 | 3.2x | var_mean on 32-64MB data, reduction overhead |

## Coordinate Descent Tuning Impact

- **40 kernels improved** (>5% faster): median 1.08x speedup, max 2.22x
- **214 neutral** (within ±5%)
- **44 worse** (<-5%): median 0.90x, worst 0.67x

CD tuning rarely helps because the dominant issue is launch overhead, not tile sizes.

## Actionable Findings

### 1. Fusion eliminates the dominant bottleneck (launch overhead)

281 of 298 real kernels reach within 2x of SOL under CUDA graph. The gap is almost entirely
per-kernel launch overhead (~30-35µs). Fusing two kernels saves one launch = ~30µs.
This validates the multi-kernel fusion analysis (see CLASSIFICATION.md).

### 2. Only 17 kernels have actual code quality issues

These are the only ones where kernel optimization (better tiling, persistent reductions,
specialized codegen) would help. The rest are already near-optimal.

### 3. var_mean and RMSNorm reductions are the main kernel-level issue

5 of 17 still-slow kernels are normalization reductions. Reduction dims of 768-1024 are
in the awkward range for GPU tiling. Persistent reductions could help here.

### 4. Tiny/scalar ops should not be kernels

16 patterns under 100KB (including 6 scalar ops). These contribute ~35µs each of pure
overhead. Candidates for CPU execution or constant folding.

### 5. TopK is inherently slow

The MoE routing topk patterns (DeepSeek-V3, Qwen3-MoE) are 4.8-5.4x even under CUDA graph.
This is an algorithmic challenge — topk requires partial sorting.
