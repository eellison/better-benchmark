# Combo Kernel Regression: Root Cause Analysis

**Target:** `sum_sum_sum_56ca14eaee84` (ViT-siglip backward, timm)
**GPU:** NVIDIA B200 (148 SMs, cc 10.0, 65536 regs/SM)
**PyTorch:** 2.13.0a0+git111efb8 (branch pr-184905)

## Summary

The regression is **NOT caused by register pressure or occupancy loss from dead code**. The actual mechanism is a **single-config autotuning failure** where the combo kernel's heuristic generates exactly one config tuned for one sub-kernel, forcing the other sub-kernel to run with an 8x-suboptimal RBLOCK.

## Measured Performance

| Configuration | Time (us) | Ratio |
|---|---|---|
| Baseline (combo_kernels=False) | 234.5 | 1.00x |
| Combo (combo_kernels=True) | 332.1 | 1.42x |
| Best possible combo config (manual) | ~206 | 0.88x |

The dominant regression is in **one combo kernel** (`triton_red_fused_0`):

| Kernel | Baseline | Combo | Ratio |
|---|---|---|---|
| kernel_0 + kernel_2 (sequential) | 92.7 + 103.3 = 196.0 us | 305.7 us (merged) | 1.56x |
| kernel_1abc + kernel_3 + kernel_4 | 7.5 + 16.7 + 14.3 = 38.5 us | 23.9 us (merged) | 0.62x |
| Final reduction | 2.5 us | 2.5 us | 1.00x |

Note: combo_1 (merging the smaller kernels) is actually **faster** than baseline. The problem is isolated to combo_0.

## Root Cause: Single-Config Heuristic for Incompatible Sub-Kernels

### The Combo Kernel Structure

`triton_red_fused_0` merges two sub-kernels:
- **Sub-A** (OUTER reduction): xnumel=196608, r=128, grid=3072 blocks (86% of total)
- **Sub-B** (DEFAULT reduction): xnumel=32768, r=768, grid=512 blocks (14% of total)

### The Config Chosen vs Optimal

| Metric | Standalone kernel_0 (Sub-A) | Standalone kernel_2 (Sub-B) | Combo kernel_0 |
|---|---|---|---|
| XBLOCK | 64 | 64 | 64 |
| R0_BLOCK | 8 | **64** | **8** |
| num_warps | 4 | **16** | **4** |
| Registers | 48 | 90 | 56 |
| Spills | 0 | 0 | 0 |
| Shared mem | 2048B | 16384B | 2048B |
| Occupancy | 62.5% | 25% | 56.2% |

**The combo uses sub-A's optimal config (RBLOCK=8, warps=4) which is terrible for sub-B.**

### Impact on Sub-B

With RBLOCK=8 instead of 64:
- Loop iterations: 768/8 * 2 loops = **192 iterations** (vs 768/64 * 2 = 24 iterations)
- That's **8x more loop iterations**

Verified experimentally (standalone kernel_2 with forced configs):

| Config | Time (us) | vs Optimal |
|---|---|---|
| XBLOCK=64, RBLOCK=64, warps=16 (optimal) | 108.7 | 1.00x |
| XBLOCK=64, RBLOCK=64, warps=4 | 124.9 | 1.15x |
| XBLOCK=64, RBLOCK=32, warps=4 | 141.1 | 1.30x |
| XBLOCK=64, RBLOCK=16, warps=4 | 153.2 | 1.41x |
| **XBLOCK=64, RBLOCK=8, warps=4 (combo's config)** | **223.0** | **2.05x** |
| XBLOCK=64, RBLOCK=128, warps=16 | 103.7 | 0.95x |

The 2.05x standalone slowdown exactly explains the combo regression.

### Why the Autotuner Picks This Config

The causal chain:

1. The combo kernel is decorated with `@triton_heuristics.reduction()` using:
   - `reduction_hint=ReductionHint.OUTER` (from sub-A)
   - `size_hints={'x': 262144, 'r0_': 128}` (from sub-A)

2. In `_reduction_configs()` (line 4215-4216 of `triton_heuristics.py`):
   ```python
   elif reduction_hint == ReductionHint.OUTER:
       return configs + [outer_config]
   ```
   **This returns EXACTLY ONE config. No autotuning happens.**

3. `outer_config_opt()` computes for x=262144, r=128:
   - x_block = 64
   - outer_r_block = 512 // 64 = 8
   - Result: (XBLOCK=64, R0_BLOCK=8)

4. This single config is compiled and used. The autotuner never tries RBLOCK=64.

### Wave Execution Analysis

With config (XBLOCK=64, R0_BLOCK=8, warps=4, 56 regs):
- Occupancy: ~9 blocks/SM
- Concurrent blocks: 148 SMs * 9 = 1332

Timeline:
```
Wave 0 (t=0 to ~31us):      1332 sub-A blocks (fast, 16 iterations each)
Wave 1 (t=~31us to ~62us):  1332 sub-A blocks (fast)
Wave 2 (t=~62us to ~306us): 408 sub-A + 512 sub-B
  - Sub-A finishes at t≈93us (16 iterations, fast)
  - Sub-B finishes at t≈306us (192 iterations, slow!)
  - 408 SMs idle for 213us waiting for sub-B to complete
```

## The Best Possible Combo Config

Running the actual combo kernel with all RBLOCK/warps combinations:

| Config | Combo Time (us) | vs Baseline (196us) |
|---|---|---|
| XBLOCK=64, RBLOCK=8, warps=4 (chosen) | 312 | 1.59x SLOWER |
| XBLOCK=64, RBLOCK=16, warps=4 | 205 | 1.05x |
| XBLOCK=64, RBLOCK=32, warps=4 | 265 | 1.35x |
| **XBLOCK=64, RBLOCK=64, warps=4** | **180** | **0.92x FASTER** |
| XBLOCK=64, RBLOCK=128, warps=4 | 306 | 1.56x |
| XBLOCK=64, RBLOCK=128, warps=8 | 199 | 1.02x |
| XBLOCK=64, RBLOCK=64, warps=16 | 242 | 1.24x |

**With RBLOCK=64, warps=4, the combo kernel is 8% FASTER than baseline!**

## Register and Occupancy Data

| Kernel | Registers | Spills | Warps | Shared | Occupancy |
|---|---|---|---|---|---|
| Baseline kernel_0 (OUTER, x=196608) | 48 | 0 | 4 | 2048B | ~62% |
| Baseline kernel_2 (DEFAULT, x=32768) | 90 | 0 | 16 | 16384B | ~25% |
| Combo kernel_0 (MERGED) | 56 | 0 | 4 | 2048B | ~56% |
| Baseline kernel_3 (OUTER) | 31 | 0 | 4 | 2048B | 100% |
| Baseline kernel_4 (DEFAULT) | 32 | 0 | 8 | 2048B | 100% |

**Key observation:** The combo kernel has FEWER registers (56) than standalone kernel_2 (90), and HIGHER occupancy (56% vs 25%). The "register pressure from dead code" hypothesis is **wrong** - registers are actually lower because RBLOCK=8 means less unrolling.

## Conclusions

### What IS the mechanism:
1. **Single-config heuristic failure**: OUTER hint returns exactly 1 config, no autotuning
2. **Conflicting RBLOCK needs**: Sub-A needs RBLOCK=8, sub-B needs RBLOCK=64
3. **Sub-A dominates config choice**: 86% of grid blocks are sub-A
4. **Long-pole effect**: Sub-B blocks take 8x longer with wrong RBLOCK, become the tail

### What is NOT the mechanism:
- Register pressure (combo has fewer regs than standalone kernel_2)
- Occupancy loss (combo has higher occupancy than standalone kernel_2)
- Dead code (both branches compile to PTX but only one executes)
- Branch overhead (trivial compared to 192 loop iterations)
- L2 cache thrashing (blocks execute sequentially, minor ~21us contribution)

### Fix recommendations:
1. **Don't merge OUTER + DEFAULT reduction kernels** - they have fundamentally incompatible tiling needs
2. **Use DEFAULT hint (multiple configs) for combo kernels** - enables autotuning
3. **Use max(r_all_subkernels) for size_hints** - ensures RBLOCK candidates cover all sub-kernels
4. **Enable per-sub-kernel RBLOCK** - the `heuristic_0`/`heuristic_1` path exists but isn't used here

### Files examined:
- Generated baseline: `/tmp/combo_investigation/baseline/torchinductor/model__0_inference_0.0/output_code.py`
- Generated combo: `/tmp/combo_investigation/combo/torchinductor/model__1_inference_1.1/output_code.py`
- Config generation: `/tmp/pytorch-work/torch/_inductor/runtime/triton_heuristics.py` (lines 4038-4256)
- Critical path: line 4215-4216 (`OUTER` hint returns single config without autotuning)
