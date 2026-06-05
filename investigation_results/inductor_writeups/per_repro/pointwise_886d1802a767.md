# pointwise_886d1802a767 — oracle_bn_relu_maxpool

## Summary
**Status**: GOOD (ratio=1.408x)
**Classification**: SCHEDULER_FUSION — realize_hint blocks fusion of pointwise into stencil consumer

## Benchmark Results
- Oracle: 34.14 us
- Compile: 48.06 us
- Ratio: 1.408x (oracle is 41% faster)

## Root Cause

Inductor generates **2 kernels**:
1. `triton_poi_fused_add_mul_reciprocal_relu_sub_unsqueeze_0` — materializes the full
   BN+ReLU activation `[1, 64, 512, 512]` float32 (64 MB) to global memory
2. `triton_poi_fused__low_memory_max_pool_with_offsets_...` — reads the 64 MB buffer
   back and computes 3x3 stride-2 maxpool with offsets

The oracle fuses both into **1 kernel** that recomputes BN+ReLU inline for each of the
9 stencil positions per output element, avoiding the 128 MB round-trip (64 MB write +
64 MB read) of the intermediate buffer.

## Why Fusion Fails

The fusion is blocked at two levels:

### 1. `realize_hint()` in lowering.py line 5560

In `/tmp/pytorch-work/torch/_inductor/lowering.py:5560`:
```python
def _max_pool_with_offsets(x, ...):
    x.realize_hint()    # <--- forces materialization
```

### 2. `realize_hint()` logic in ir.py line 10103-10111

In `/tmp/pytorch-work/torch/_inductor/ir.py:10103`:
```python
def realize_hint(self) -> None:
    if (
        isinstance(self.data, (Pointwise, Reduction))
        and self.data.inner_fn_opcount().nontrivial_read_count > 1
    ):
        self.realize()
```

The BN+ReLU pointwise has `nontrivial_read_count = 5` (conv + mean + var + weight + bias),
so it triggers materialization. However, 4 of those 5 reads are from tiny [64]-element
broadcast tensors that fit in L1 cache and are essentially free to recompute.

### 3. Why the heuristic is wrong here

The `nontrivial_read_count > 1` threshold does not distinguish:
- **Large tensor reads** with diverse indices (expensive to recompute due to memory traffic)
- **Small broadcast reads** that are L1-resident and trivially cheap to recompute

For this workload:
- Intermediate buffer: 64 MB (write) + 64 MB (read back) = 128 MB traffic
- Broadcast params: 4 * 64 * 4 bytes = 1 KB total (fits in a single cache line set)
- Recompute cost: 4 ALU ops (sub, mul, add, relu) per element * 9 stencil positions

The recompute is overwhelmingly cheaper than the memory round-trip.

## Kernel Count
- Inductor: 2 kernels (pointwise BN+ReLU + maxpool stencil)
- Oracle: 1 kernel (fused BN+ReLU+maxpool)

## Config Exploration
- `coordinate_descent_tuning = True`: no impact (2 kernels still)
- `combo_kernels = True`: no impact (can't combine different grid shapes)
- `combo_kernel_per_subkernel_blocks = True`: no impact
- `triton.multi_kernel = 3`: no impact (this is about reduction variants, not fusion)
- `max_autotune = True`: no improvement

## Design Doc: Proposed Fix

### Option A: Smart realize_hint for broadcast-dominated producers (preferred)

Modify `realize_hint()` in `ir.py` to account for the *size* of reads, not just
their count. A pointwise node whose non-trivial reads are mostly from small
broadcast tensors (numel << output numel / stencil_factor) should NOT be realized.

**File**: `/tmp/pytorch-work/torch/_inductor/ir.py:10103-10111`

```python
def realize_hint(self) -> None:
    if isinstance(self.data, (Pointwise, Reduction)):
        opcount = self.data.inner_fn_opcount()
        if opcount.nontrivial_read_count > 1:
            # Don't realize if most reads are small broadcast tensors
            read_sizes = [V.graph.get_dep_size_hint(dep) for dep in self.get_reads()]
            output_size = V.graph.get_dep_size_hint(self)  # or product of ranges
            large_reads = sum(1 for s in read_sizes if s > output_size / 16)
            if large_reads > 1:
                self.realize()
```

### Option B: Remove realize_hint from pool lowering (simpler but broader impact)

Remove `x.realize_hint()` from `_max_pool_with_offsets` in `lowering.py:5560` and
let the scheduler decide based on memory traffic analysis. The scheduler already has
the information about producer/consumer sizes to make this decision.

**Risk**: This could cause excessive recomputation for complex producers consumed
by large stencils. Would need careful benchmarking across the pool-consuming workloads.

### Option C: Scheduler-level fusion of pointwise into stencil consumers

Add a scheduler pass that detects when a pointwise producer's recomputation cost
(ops * stencil_size) is cheaper than the materialization cost (2 * buffer_bytes / bandwidth).
This is the most principled fix but requires the most engineering.

**File**: `/tmp/pytorch-work/torch/_inductor/scheduler.py` (fusion scoring)

## Affected Repros
This pattern (BN/affine normalization + ReLU fused with maxpool stencil) is common in:
- ResNet, VGG, and similar CNN architectures with BN->ReLU->MaxPool sequences
- `torchbench_doctr_det_predictor` (this repro's source)
- Any model with `nn.BatchNorm2d -> nn.ReLU -> nn.MaxPool2d`

The existing writeup for `pointwise_849c8a7e937a` (oracle_bn_affine_residual) likely
shares the same underlying realize_hint issue for affine normalization patterns.
