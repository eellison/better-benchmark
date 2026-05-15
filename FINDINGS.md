# PyTorch Inductor Kernel Performance — B200 Investigation

## Implemented Fixes

### 1. Persistent Reduction Threshold (INNER): 1024 → 4096

**File:** `torch/_inductor/choices.py:401`

The INNER persistent reduction threshold was too conservative for B200 (65536 regs/SM, 2048 threads/SM). Raising from 1024 to 4096 allows layer_norm, softmax, and variance reductions to stay in registers instead of spilling to a loop.

**Impact:** 169 kernels across 14+ models now use persistent reduction.

| Config | Before | After | Speedup |
|--------|-------:|------:|--------:|
| GPTNeo-like (4096×2048, 24L) | 310 us | 184 us | 1.68x |
| Large batch (8192×2048, 12L) | 296 us | 152 us | 1.95x |
| Blenderbot-like (2048×4096, 12L) | 123 us | 81 us | 1.51x |
| DebertaV2-like (1024×1536, 24L) | 120 us | 88 us | 1.36x |

### 2. Persistent Reduction Threshold (OUTER): 64 → 128

Explicit OUTER threshold of 128 helps 165 training gradient kernels (feature-dim accumulations with r=128, large x). ~1.22-1.28x speedup for x >= 49152.

### 3. DEFAULT Persistent Reduction Threshold: 64 → 1024

Fixes a 3.2-3.5x gap on gpt-oss-20b's 513-dim max kernel. The non-power-of-2 dimension (512+1 sink token) caused catastrophic tiling: 2 loop iterations where the second processes 1/512 useful elements. Making it persistent eliminates the loop entirely. Coordinate descent tuning independently confirmed 2.22x speedup for this kernel.

**Tests:** 928 GPU tests pass. Threshold=4096 avoids numerical tolerance issues seen at 8192 (one test accumulates softmax over r=4190 through embedding_bag+var_mean chain).

---

## Optimization Opportunities Found (Not Yet Implemented)

### A. GQA Q/K RoPE Missed Fusion — 3 kernels where 1 suffices

**Impact:** 5 of 12 vLLM multi-kernel repros. Every GQA attention layer (Mistral-7B, Qwen3, gpt-oss-20b).

**Problem:** Q and K projections come from independent matmuls. Each goes through RoPE independently. With GQA, K also goes through `repeat_kv` (expand+clone). Inductor generates 3 kernels:
- K RoPE [B, H_kv, S, D] — 2M elements
- K expand+clone [B, H_q, S, D] — 8M elements  
- Q RoPE [B, H_q, S, D] — 8M elements

Fusion fails because: no shared data (cos/sin are small scalar broadcasts, not tracked).

**Ideal:** 1 kernel reading both Q and K, writing both RoPE'd outputs. Benchmarked at 2.25x faster than inductor's 2-kernel approach.

**Fixes:**
1. *transformers level:* Pass `is_causal=True` instead of materializing attention_mask → enables `enable_gqa=True` → eliminates expand+clone entirely
2. *SDPA:* Support GQA in math backend (currently only flash/mem_efficient)
3. *inductor:* Multi-output fusion — relax "shared data" requirement when kernels have same grid shape
4. *inductor:* Multi-store codegen for expand+clone (iterate producer domain, emit N stores)

**Repros:** `output/investigations/02_gqa_qk_rope_fusion.md`

---

### B. DeepSeek-V3 Interleaved RoPE + 128x Broadcast — 3.73x gap

**Impact:** 1 kernel, 370MB, 221us (sol=59us). DeepSeek-V3 specific.

**Problem:** Three compounding issues:
1. Interleaved permute `[0,1,2,4,3]` + clone — stride-2 cache-unfriendly access
2. 128x K broadcast expand — reads same K data 128 times (256KB → 32MB of reads)
3. Two cat ops — conditional branching in inner loop

**Fixes:**
1. Multi-store for K broadcast (compute once, write 128x)
2. Shared memory for interleave permute
3. Direct output indexing for cat (compute out_idx → source_idx at compile time)

**Repro:** `output/aten_repros/vllm_deepseek-ai_DeepSeek-V3_inference/region_012_pointwise_c0b2590bfdf4_d2f5d45b.py`

---

### C. repeat_kv expand+clone Bandwidth Waste — 1.7-2.1x gap

**Impact:** 3 standalone kernels + appears in 6/12 multi-kernel repros as fusion barrier.

**Problem:** GQA models expand K/V from `[B, H_kv, S, D]` to `[B, H_q, S, D]` via expand+clone. Inductor iterates the consumer domain, reading each element N_rep times.

| Model | Expand | Bytes | graph_us | sol_us | Gap |
|---|---|---|---|---|---|
| Mistral-7B | 4x | 21MB | 8.3 | 4.0 | 2.08x |
| Qwen3-30B-A3B | 8x | 19MB | 8.3 | 4.6 | 1.78x |
| gpt-oss-20b | 8x | 38MB | 8.3 | 5.0 | 1.67x |

**Fix:** Multi-store codegen: iterate producer domain [B, H_kv, S, D], write to N_rep output locations.

---

### D. TopK Comparison-Bound (MoE Routing) — 1.8-5.4x gap

**Impact:** 4 kernels in MoE models (Qwen3-MoE, DeepSeek-V3, gpt-oss-20b).

**Problem:** These kernels are comparison/sort-bound, not memory-bound. The memcopy SOL metric is fundamentally wrong for them. They're doing O(n*k) comparisons in the TopK selection.

| Model | Inner Dim | k | graph_us | sol_us | Gap |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | 128 | 8 | 25.9 | 4.8 | 5.44x |
| DeepSeek-V3 | 32 (8 groups) | 2 | 43.1 | 9.0 | 4.79x |
| gpt-oss-20b | 32 | 4 | 14.4 | 4.6 | 3.12x |

**Fix:** Algorithmic improvement in TopK (partial sort / bitonic sort instead of full comparison). Or: accept that these are compute-bound and SOL is the wrong metric.

---

### E. Welford var_mean Compute-Bound — 1.6-3.1x gap (misleading metric)

**Impact:** 12 kernels across all LayerNorm models.

**Problem:** These are NOT underperforming. They do GELU_tanh + two-pass reduction (Welford). The gap is inherent ALU cost — memcopy SOL is the wrong baseline. Already using persistent reduction.

**Verdict:** No fix needed. Reclassify as compute-bound; don't track against memory SOL.

---

### F. Constant Mask Generation — 1.5-2.5x gap (misleading metric)

**Impact:** 3 kernels (OPT, Bert, Roberta).

**Problem:** Causal mask computed from `full + iota + triu + where + expand`. These kernels have zero real tensor inputs — they generate data from constants. SOL based on output bytes is misleading because compute (triu branching) dominates.

**Verdict:** Low priority. Could be hoisted to a constant buffer at graph level.

---

## Methodology

### Data Collection
1. **Debug traces:** Ran 32 HuggingFace models (inference+training) with `TORCH_COMPILE_DEBUG=1`, parsed `output_code.py` for kernel metadata → 2495 kernel index
2. **Post-grad subgraph extraction:** `inductor_config.post_grad_custom_pre_pass` captures FX GraphModule, partitions with `is_fusible_node`, extracts minimal subgraphs → 1860 standalone repros (hashed by `pattern_ops × input_shapes`)
3. **Multi-kernel detection:** Each repro compiled independently; regions generating >1 kernel flagged as potential missed fusions → 141 training + 59 inference cases
4. **SOL benchmarking:** Each kernel benchmarked against memcopy at matching transfer size, with CUDA graph replay to eliminate launch overhead

### Key Insight on Fusion
After investigating all 141+59 multi-kernel cases, the "missed fusions" fall into 3 categories:
1. **Weight-sharing accumulation** (ALBERT 28-kernel case): 12 layers share one weight, backward accumulates independently — correctly separate
2. **Different reduction dimensions** (cross-entropy): amax(dim=1) + sum(dim=1) → sum.default — correctly separate (different iteration spaces)
3. **No shared data** (Q/K RoPE, expand+clone): independently computed buffers — correctly separate per current scheduler policy, but a real optimization opportunity (items A-C above)

### Hardware
- NVIDIA B200: 148 SMs, cc=100, 65536 regs/SM, 2048 threads/SM, ~8 TB/s HBM
- MAX_R0_BLOCK = 1024 for Blackwell (already tuned)
