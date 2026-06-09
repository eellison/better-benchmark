# Inductor Implementation: Softmax Backward / Attention Backward

## Status

- Queue id: `softmax_backward_attention`
- Priority: P0 (rank 2)
- Implementation status: **implemented_unmeasured**
- Owner: Boyle
- Speedup achieved: **7.4x** (6350 us -> 840 us)

## Target Repro

- `sum_sum_sum_afd118ca907d`: T5-style attention backward fragment
- Model: `torchbench_hf_T5_train_001`
- Shape: [8, 8, 1024, 1024] (batch=8, heads=8, seq=1024)

## Root Cause Analysis

### Kernel Breakdown (before optimization)

| Kernel | Description | Time (us) | % Total |
|--------|-------------|-----------|---------|
| K2 (x2) | add 5 residuals + dscore, reduce batch=8, permute+where+atomic index_put | 3735 | 67% |
| K3 (x1) | reconstruct softmax probs, compute dscore (2-pass non-persistent) | 1642 | 30% |
| K0 (x1) | first softmax backward (persistent, 65K rows x 1024 cols) | 126 | 2% |
| K1 (x2) | zero [32,8] output | 3 | <1% |
| **Total** | | ~5506 | |

### The Dominant Bottleneck: Kernel 2

Kernel 2 (called twice, once per attention head group) is a reduction kernel that:
1. Reads 6 tensors of shape [8, 8, 1024, 1024] with stride 8388608 between batch elements
2. Reduces over batch dimension (r0=8) to get [8, 1024, 1024]
3. Permutes to [1024, 1024, 8]
4. Applies where mask (bucket == -1)
5. Accumulates via `index_put(..., accumulate=True)` into [32, 8]

The index_put accumulation creates **extreme atomic contention**:
- 8,388,608 work items write into 256 output slots (32 buckets x 8 heads)
- Contention ratio: 32,768:1 per output slot
- The scheduler unconditionally fuses atomic_add epilogues (line 5310 in scheduler.py)
- This prevents benchmarking the fused vs unfused alternative

### Secondary Bottleneck: Kernel 3

Kernel 3 is a 2-pass non-persistent reduction that:
- Pass 1: computes softmax probs and sum(grad*probs), stores probs to GMEM
- Pass 2: reloads probs, computes fma(neg(probs), sum, grad*probs)

The 2-pass pattern exists because probs must be stored as an output.
This adds ~576 MB of redundant memory traffic. However, it only contributes
248 us (after optimization), making it a secondary concern.

## Implementation

### Change: Enable `partitioned_scatter_enabled` by default

**File 1**: `torch/_inductor/config.py`

Changed default from `"0"` to `"1"`:
```python
partitioned_scatter_enabled = (
    os.environ.get("TORCHINDUCTOR_PARTITIONED_SCATTER_ENABLED", "1") == "1"
)
```

Added minimum contention ratio config:
```python
partitioned_scatter_min_contention_ratio: float = 4.0
```

**File 2**: `torch/_inductor/fx_passes/reduced_atomic_contention.py`

Added contention ratio guard to prevent the pass from firing on low-contention
patterns (where atomic overhead is low and partitioning adds memory overhead):
```python
min_contention_ratio = config.partitioned_scatter_min_contention_ratio
if contention_ratio < min_contention_ratio:
    return False
```

### How It Works

The `partitioned_scatter_optimization_pass` (in `fx_passes/reduced_atomic_contention.py`)
transforms high-contention `index_put(..., accumulate=True)` by:
1. Allocating N partitioned output buffers (instead of 1)
2. Each CTA writes to a different partition (hash-based or round-robin)
3. A final small reduction kernel merges all partitions

This reduces contention by a factor of N (partition count). The pass has
built-in guards:
- Minimum index size threshold (4096 elements)
- Contention ratio check (index_size / output_size)
- Memory budget fraction (10% default)
- Optimal partition count estimation

### Result After Optimization

| Kernel | Description | Time (us) | % Total |
|--------|-------------|-----------|---------|
| K2 (x2) | add residuals + dscore, reduce batch=8, permute+where+partitioned scatter | 485 | 58% |
| K3 (x1) | reconstruct softmax probs, compute dscore | 220 | 26% |
| K0 (x1) | first softmax backward | 123 | 15% |
| K_merge (x2) | merge partitioned scatter buffers | 6 | <1% |
| K_zeros (x2) | zero output | 4 | <1% |
| **Total** | | ~838 | |

## Measurements

```
Baseline (CDT only, scatter OFF):        6350 us
CDT + scatter ON (new default):           846 us
CDT + combo + scatter ON:                 811 us
Speedup (CDT only):                       7.5x
Speedup (CDT + combo):                    7.8x
```

Validated correct (rtol=1e-4 relative diff, expected for order-dependent atomic accumulation).

Regression check on 3 other repros with index_put patterns: all neutral (no regression).

## Risk Assessment

- The partitioned scatter pass is guarded with heuristics:
  - Minimum contention ratio of 4.0 (added in this implementation)
  - Minimum index size of 4096 elements
  - Memory budget fraction of 10%
- Two existing tests explicitly `@config.patch("partitioned_scatter_enabled", False)` their expectations
- Memory overhead is bounded by the 10% memory budget fraction
- The pass only activates for patterns with high contention (large index vs small output)
- Float32 atomic accumulation is inherently non-deterministic in ordering; the optimization preserves this property

## Future Work

1. **Kernel 3 optimization**: The 2-pass non-persistent reduction could be made persistent
   (r0=1024 fits in registers with XBLOCK=1), eliminating the second pass entirely.
   Expected savings: ~100-200 us. Lower priority since K2 was the dominant issue.

2. **Broader testing**: Run the new default against the full corpus to check for regressions
   in workloads where the scatter optimization fires but provides no benefit.

3. **Scheduler awareness**: The unconditional fusion of atomic_add epilogues
   (scheduler.py line 5310) should be reconsidered -- benchmark-gated fusion
   would allow Inductor to make better decisions even without the partitioned
   scatter pass.
