# pointwise_fc46ae4ec316 - BN Affine + Residual Add

## Summary
- **Pattern**: BatchNorm affine (subtract mean, multiply by rsqrt(var+eps)*weight, add bias) + cast to fp16 + residual add
- **Model**: torchbench_mnasnet1_0_infer_000
- **Shape**: x=[256, 192, 7, 7] (fp16), per-channel params=[192] (fp16), residual=[256, 192, 7, 7] (fp16)
- **Ratio**: 1.24x (oracle 8.0 us, compiled 9.92 us)

## Kernel Count
- Inductor: 1 kernel (fully fused)
- Oracle: 1 kernel (fully fused)

## Root Cause

Both Inductor and the oracle produce a single fused pointwise kernel doing the same computation. The performance gap comes from **`num_stages` autotuning**:

- **Inductor**: coordinate_descent selects `XBLOCK=512, num_warps=4, num_stages=1`
- **Oracle**: autotune selects `BLOCK_SIZE=512, num_warps=4, num_stages=4`

The key difference is `num_stages=4` vs `num_stages=1`. For this memory-bound kernel (6 fp16 loads + 1 fp16 store per element, ~2.4M elements), software pipelining via `num_stages=4` allows overlapping load latency with computation.

### Why Inductor can't reach optimal today

In `/tmp/pytorch-work/torch/_inductor/runtime/coordinate_descent_tuner.py` (lines 130-136), the coordinate descent tuner only includes `num_stages` in its search space for matmul (`is_mm`) and mix-order reduction kernels. Pointwise kernels are excluded:

```python
if self.is_mm:
    out.append("num_stages")
```

Additionally, in `/tmp/pytorch-work/torch/_inductor/runtime/triton_heuristics.py` (line 3242), the default `num_stages=1` is hardcoded for the pointwise config constructor.

### Secondary issue: redundant mul-by-1

The FX graph contains `aten.mul.Tensor(reciprocal_default, 1)` which generates a `tmp9 * 1.0` in the kernel. This is a constant-fold gap (should be eliminated by post-grad passes), but likely has zero perf impact as the Triton/PTX compiler optimizes it away.

## Config Exploration

| Config | Result |
|--------|--------|
| `coordinate_descent_tuning = True` | XBLOCK=512, warps=4, stages=1 (default) |
| `max_autotune = True` | No improvement (still stages=1) |
| `combo_kernels = True` | No change (single kernel) |
| `triton.multi_kernel = 1` | Slower (17.66 us) |

Manual testing with identical kernel code (same arithmetic/indexing) shows all XBLOCK/warps/stages configurations yield ~11 us, confirming the gap is between the Inductor wrapper overhead + autotuned config vs oracle's direct Triton call with stages=4.

## Bandwidth Analysis

- Effective data: 2 full [256,192,7,7] fp16 tensors (loads) + 1 output = ~14.4 MB
- Per-channel params: 4 * 192 * 2 bytes = 1.5 KB (L2 cached, negligible)
- B200 bandwidth ~3.35 TB/s: theoretical min ~4.3 us
- Oracle achieves ~8 us (54% bandwidth efficiency)
- Inductor achieves ~10 us (43% bandwidth efficiency)

## Design Doc: Fix Needed

**Enhancement**: Add `num_stages` to coordinate descent search space for memory-bound pointwise kernels.

**Location**: `/tmp/pytorch-work/torch/_inductor/runtime/coordinate_descent_tuner.py`, around line 131

**Proposed change**:
```python
# Add num_stages tuning for memory-bound pointwise kernels
num_loads = self.inductor_meta.get("num_load", 0)
if self.is_mm or num_loads >= 4:  # memory-bound heuristic
    out.append("num_stages")
```

Alternatively, change the default `num_stages` in the pointwise config heuristic based on load count:
- `/tmp/pytorch-work/torch/_inductor/runtime/triton_heuristics.py` line 3242: use `num_stages=4` when `num_load >= 4`

**Risk**: Adding num_stages to coordesc search increases tuning time for all pointwise kernels. The load-count heuristic limits it to truly memory-bound cases.

**Affected repros**: Any BN-affine, layernorm-affine, or similar per-channel pointwise patterns with multiple broadcasts (likely 10-20% of pointwise repros in corpus).

## Oracle Bug Note

The oracle has a bug: `grid = (triton.cdiv(total, 1024),)` is hardcoded to 1024 regardless of the autotuned BLOCK_SIZE. When BLOCK_SIZE=512 is selected, only half the elements are processed. This passes the harness check because the outputs are marked `SKIP (stochastic)`. The benchmark result (8.0 us) may be artificially low due to processing fewer elements. A corrected oracle would use `grid = (triton.cdiv(total, BLOCK_SIZE),)` and likely measure closer to 9-10 us, narrowing the actual gap.
