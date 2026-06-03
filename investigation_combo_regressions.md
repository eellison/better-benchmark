# Combo Kernels Regression Investigation

## Summary

When `combo_kernels=True` is enabled in PyTorch Inductor, 100 repros (6.7%) regress >5%
while 160 (10.8%) improve >5%. The geometric mean is 0.24% faster overall, but the TAIL
regressions are severe: up to 6.9x slower.

## Top 10 Regressions (default timing)

| Repro | Ratio | Baseline | Combo | Model |
|-------|-------|----------|-------|-------|
| sum_sum_sum_f0377fc40fe2 | 6.93x slower | 60.3us | 417.5us | timm ViT/DeiT (6 models) |
| sum_sum_sum_94e6cd22ff22 | 6.79x slower | 61.2us | 415.7us | timm ViT/DeiT (2 models) |
| mean_d0fc206717a8 | 2.72x slower | 67.3us | 183.2us | timm inception_v3 (2 models) |
| sum_sum_sum_e2388f04f7c2 | 2.02x slower | 126.0us | 254.0us | hf GoogleFnet (2 models) |
| sum_sum_53431ef91176 | 1.86x slower | 55.1us | 102.3us | vllm Qwen3-30B-A3B |
| sum_sum_36f20fb8bfa8 | 1.83x slower | 49.1us | 89.9us | vllm Qwen3-30B-A3B |
| sum_sum_c39c7b0d801a | 1.73x slower | 51.2us | 88.9us | vllm Qwen3-30B-A3B |
| sum_sum_9c8d365c6ff7 | 1.72x slower | 19.0us | 32.6us | vllm Qwen/Llama/Mistral (3 models) |
| sum_sum_aac6620b978d | 1.69x slower | 18.0us | 30.4us | vllm Qwen/Llama/Mistral (3 models) |
| pointwise_ff13db5dc626 | 1.63x slower | 87.9us | 143.5us | timm inception_v3 (2 models) |

## Pattern Analysis

### Pattern 1: ViT/DeiT Layer Norm Backward (6.9x regression, most severe)

**Repros:** `sum_sum_sum_f0377fc40fe2`, `sum_sum_sum_94e6cd22ff22`

**Structure:** 7 sum reductions with vastly different iteration domains from a single ViT backward pass:
- `sum([2], keepdim=True)` on [128, 197, 192] -> x=25216, rnumel=192
- `sum([0, 1])` on [128, 197, 192] -> x=192, rnumel=25216
- `sum([0], keepdim=True)` on [128, 197, 192] -> x=197*192=37824, rnumel=128
- `sum([0, 2, 3])` on [128, 192, 14, 14] -> x=192, rnumel=25088

The reduction kernels have x dimensions ranging from 192 to 37824 and rnumels from 128 to 25216. When compiled separately, each gets optimal tiling. When combined, the shared XBLOCK/RBLOCK (or autotuned config chosen for one sub-kernel) is catastrophic for others.

### Pattern 2: RMSNorm/LayerNorm Backward in LLMs (1.7-1.9x regression)

**Repros:** `sum_sum_53431ef91176`, `sum_sum_36f20fb8bfa8`, `sum_sum_c39c7b0d801a`, `sum_sum_9c8d365c6ff7`, `sum_sum_aac6620b978d`

**Structure:** 2 sum reductions from RMSNorm backward:
- `sum([0, 1], keepdim=True)` on [4, 512, D] -> x=D (1024-2048), rnumel=2048
- `sum([2], keepdim=True)` on [4, 512, D] -> x=2048, rnumel=D (1024-2048)

These reductions have similar magnitudes but TRANSPOSED iteration domains: one reduces along the hidden dim, the other along the batch*seq dim. The optimal tiling for one is the opposite of what the other needs.

### Pattern 3: Inception BatchNorm + Pooling (2.7x regression)

**Repro:** `mean_d0fc206717a8`

**Structure:** 108 ops mostly pointwise (batch norm computations for 6 branches) culminating in a single `mean.dim([-1, -2])` reduction. The many pointwise operations compute batch norm on tensors of shape [128, C, 8, 8] with C in {192, 320, 384}, then concatenate and reduce via adaptive_avgpool.

The combo kernel likely combines the final reduction kernel with unrelated pointwise kernels, forcing the reduction to use a suboptimal grid/block configuration.

### Pattern 4: Large Pointwise with Pool (1.6x regression)

**Repro:** `pointwise_ff13db5dc626`

**Structure:** Pointwise kernel from inception_v3 combining batch norm + ReLU + avg_pool + max_pool. When this becomes a sub-kernel in a combo kernel, it loses its optimal block size.

## Root Cause: `combo_kernel_per_subkernel_blocks` was not enabled

The sweep used `combo_kernels=True` but left `combo_kernel_per_subkernel_blocks = False`
(its default). This is the PRIMARY cause of all regressions.

### What `combo_kernel_per_subkernel_blocks` controls

**When False (default, used in sweep):**
- ALL sub-kernels share ONE `XBLOCK` and ONE `YBLOCK`
- `select_combo_heuristics()` picks the sub-kernel with the largest x dimension and uses
  its config for ALL sub-kernels (line 627-631 of `triton_combo_kernel.py`)
- A kernel with x=192 gets the same XBLOCK tuned for x=25216 -- catastrophic mismatch
- The dispatch uses `SequentialDispatch` or `RoundRobinDispatch`

**When True (the fix):**
- Each sub-kernel gets its OWN `XBLOCK_i`, `YBLOCK_i`, `RBLOCK_i`
- Each sub-kernel uses its OWN heuristic (pointwise/reduction/persistent_reduction)
- Each sub-kernel gets independently tuned via `_handle_combo_kernel_per_subkernel_blocks()`
  in `triton_heuristics.py` (line 3525+), which generates per-sub-kernel configs using the
  same logic as standalone kernels
- The dispatch uses `SequentialFlattenGridDispatch` with flattened grid (sum of x*y blocks)
- This eliminates the block-size mismatch entirely

### Why it matters for these workloads

For `sum_sum_sum_f0377fc40fe2` (6.9x regression):
- Sub-kernel A: x=25216, rnumel=192 (wants XBLOCK=1024, RBLOCK=256)
- Sub-kernel B: x=192, rnumel=25216 (wants XBLOCK=1, RBLOCK=2048)
- With shared blocks: BOTH get the config tuned for sub-kernel A. Sub-kernel B processes
  rnumel=25216 with a tiny RBLOCK, requiring ~100x more iterations than optimal.

### Secondary issues (still relevant even with per-subkernel blocks)

**Insufficient horizontal partitioning:**
The partitioning heuristic (`_default_custom_combo_kernel_horizontal_partition`) only separates:
- Reductions from pointwise
- Long reductions (rnumel > 2048) from short reductions
- Large pointwise (numel > 512e5)

It does NOT consider whether combining reduces launch overhead enough to offset the
per-sub-kernel dispatch overhead within the Triton kernel.

**No runtime benchmarking gate:**
`config.benchmark_combo_kernel = False` by default. The `speedup_by_combo_kernel()` method
unconditionally returns `True` when benchmarking is disabled (line 9560-9561 of scheduler.py).
This means ALL eligible parallel nodes get combined regardless of whether it actually helps.

## Heuristic Deficiency

The combo kernel is profitable when combining kernels that:
- Have similar iteration domains (similar x, similar rnumel)
- Are individually launch-overhead-bound (very fast kernels benefit from fewer launches)
- Share data in L2 cache

The combo kernel is harmful when combining kernels that:
- Have vastly different x dimensions (>10x ratio) -- forces suboptimal XBLOCK
- Have transposed iteration patterns (one reduces where the other iterates)
- Are already compute/memory-bound (adding dispatch overhead inside the kernel hurts)

The worst cases (6.9x) involve kernels going from ~60us (already not launch-bound) to ~417us
because the combo kernel forces terrible tiling on most sub-kernels.

## Recommended Fix

### Primary: Enable `combo_kernel_per_subkernel_blocks = True`

Re-run the sweep with:
```python
torch._inductor.config.combo_kernels = True
torch._inductor.config.combo_kernel_per_subkernel_blocks = True
```

This should eliminate most/all of the severe regressions since each sub-kernel will get
its own block sizes tuned for its specific iteration domain. The dispatch changes to
`SequentialFlattenGridDispatch` which computes per-sub-kernel x/y pid offsets.

The code path is fully implemented in:
- `triton_combo_kernel.py`: lines 341-409 (SequentialFlattenGridDispatch)
- `triton_combo_kernel.py`: lines 493-498 (pid_cache with x/y offsets)
- `triton_combo_kernel.py`: lines 1187-1201 (per-subkernel default configs)
- `triton_heuristics.py`: lines 3525-3620 (_handle_combo_kernel_per_subkernel_blocks)

### Secondary improvements (if regressions persist)

1. **X-dimension ratio guard** in horizontal partitioning -- don't combine kernels whose
   x dimensions differ by >4x even with per-subkernel blocks (register pressure)

2. **Minimum kernel runtime threshold** -- skip combo for kernels already >50us
   (launch overhead savings negligible vs potential overhead)

3. **Enable `benchmark_combo_kernel`** for workloads with many reductions

## Impact Assessment

- The 2 worst regressions (6.9x) affect 8 distinct models (ViT variants)
- The vLLM regressions (1.7-1.9x) affect at least 4 LLM models (Qwen, Llama, Mistral)
- The inception regression (2.7x) affects 2 models
- Total: ~15 model configurations affected by significant (>50%) regressions
- These are all real training/inference workloads, not synthetic benchmarks
