# Combo Kernel Regressions: Transposed Reduction Domains

## Summary

When `combo_kernels=True` + `combo_kernel_per_subkernel_blocks=True`, the feature achieves a **1.015x geomean speedup** across 1482 corpus repros (205 wins >5%, 4 confirmed regressions >15%). The regressions all share a single pattern: LayerNorm/GroupNorm backward ops that produce parallel reductions with **transposed iteration domains**.

## The Pattern

LayerNorm backward computes two independent reductions on the same tensor:
- `sum(grad, dim=hidden)` → xnumel=batch, rnumel=hidden
- `sum(grad, dim=batch)` → xnumel=hidden, rnumel=batch

These are parallel (no data dependency) and get scheduled as separate kernels. The combo kernel merger sees them as "same-category reductions" and combines them into one dispatch. But their iteration domains are **inverted** — one wants large XBLOCK/small RBLOCK, the other wants the opposite.

## Confirmed Regressions

| Repro | Model | Baseline (sum of parts) | Combo | Overhead | x-ratio |
|-------|-------|------------------------|-------|----------|---------|
| `sum_sum_sum_56ca14eaee84` | ViT-siglip-256 | 197 us | 388 us | **1.97x** | 6x (196608 vs 32768) |
| `sum_sum_sum_6a2ad206248c` | ElectraForCausalLM | 84 us | 152 us | **1.81x** | 256x (32768 vs 128) |
| `sum_sum_sum_82a3f0084247` | Swin Transformer | 715 us | 860 us | **1.20x** | 3136x (401408 vs 128) |
| `sum_sum_sum_e2388f04f7c2` | GoogleFnet | 121 us | 143 us | **1.18x** | 21x (16384 vs 768) |

## Why It's Slow

The combo kernel compiles all sub-kernels into a single Triton binary. Each threadblock dispatches to one sub-kernel via `pid`-based branching. The problems:

1. **Register pressure from dead code.** Every block carries register allocations for ALL sub-kernels' live variables, even though only one branch executes. A sub-kernel needing 40 registers for OUTER reduction + a sub-kernel needing 48 registers for DEFAULT reduction = the compiler allocates ~88 registers per thread → occupancy drops from 50% to 25%.

2. **Incompatible reduction strategies.** The baseline picks different strategies per kernel:
   - `x=196608, r=128` → OUTER hint, persistent reduction, R0_BLOCK=128 (no loop)
   - `x=32768, r=768` → DEFAULT hint, split reduction, large RBLOCK

   In the combo, both live in one binary. `per_subkernel_blocks` gives them separate tile sizes, but they still share register allocation and the compiler can't apply persistent-reduction optimizations to a single branch of a multi-branch kernel.

3. **Sequential grid dispatch overhead.** The combo grid is `sum(ceil(xnumel_i / XBLOCK_i))` total blocks. Each block executes a `pid < threshold` comparison chain. For the large sub-kernel (196608/XBLOCK blocks), this branch overhead is paid per-block and is pure waste — it would never have existed in the standalone kernel.

## Why `per_subkernel_blocks` Doesn't Fix It

`per_subkernel_blocks` gives each sub-kernel its own XBLOCK/RBLOCK/num_warps. This helps when sub-kernels have similar xnumel but different optimal tile sizes. It does NOT help when:
- The register pressure comes from having multiple code paths in one binary (architectural, not tunable)
- One sub-kernel wants persistent reduction (no loop) while another needs a loop
- The xnumel ratio means one sub-kernel has 1000x more blocks than another

## Proposed Fix: xnumel-Ratio Guard

In `_default_custom_combo_kernel_horizontal_partition` (triton_combo_kernel.py), add a guard that prevents combining reductions whose xnumel values differ by more than 8x:

```python
def _default_custom_combo_kernel_horizontal_partition(nodes, node_info_map, ...):
    # ... existing partitioning ...
    
    # Within short-reduction group, further split by xnumel magnitude
    # Don't combine kernels whose xnumel differs by more than 8x
    xnumel_buckets = defaultdict(list)
    for n in short_reduction_group:
        xnumel = get_xnumel(node_info_map[n])
        bucket = int(math.log2(max(xnumel, 1))) // 3  # groups within 8x of each other
        xnumel_buckets[bucket].append(n)
    
    for bucket_nodes in xnumel_buckets.values():
        partitions.append(bucket_nodes)
```

This would:
- Fix all 4 severe regressions (all have x-ratio >8x)
- Not affect the 205 wins (which combine same-shape pointwise ops or reductions with similar xnumel)
- Cost nothing at runtime (partition decision is at compile time)

## Alternative Fixes

**Reduction hint compatibility check.** Don't combine OUTER with DEFAULT/INNER reductions. More conservative — might over-split some beneficial cases.

**Runtime threshold.** Don't combo sub-kernels whose standalone runtime >50us. At 50us, kernel launch overhead is ~10% of runtime, so combo can save at most 10% — but its dispatch overhead is typically 5-20%, making it net negative for large kernels.

**Benchmark gate.** Set `config.benchmark_combo_kernel = True` for reduction kernels. This runtime-benchmarks the combo vs standalone and picks the winner. Most correct, but doubles compile time for affected kernels.

## Measurement Notes

- 61 of the 67 sweep-indicated regressions were **measurement noise** (confirmed via live A/B benchmarking)
- All noise cases were single-kernel repros where kernel count didn't change (combo never fired)
- Enabling combo_kernels can trigger different scheduler ordering, but this doesn't affect codegen quality
- Interleaved min timing (5 rounds, single GPU lock hold) eliminates these false positives
