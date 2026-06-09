# pointwise_9903e70019c2 (Longformer Sliding-Window Layout Stencil)

## Benchmark Result
- Oracle: 9.63 us
- Compiled: 11.94 us
- Ratio: 1.239x
- Status: Regression

## Root Cause

The oracle classifies this as SCHEDULER_FUSION. The repro computes:
1. div by 8.0 on f16[4096, 768] input (via views/permutes this becomes f16[12,4096,64])
2. View as f16[12, 8, 512, 64]
3. as_strided with overlapping windows: [12, 15, 512, 64] stride [64, 196608, 768, 1] (Longformer sliding window)
4. Clone to contiguous format
5. View to [180, 512, 64]

The oracle fuses the div and the overlapping stencil materialization into a single kernel: it reads from the original input with the correct stencil indexing (window*window_step + pos) and multiplies by 0.125 inline.

Inductor generates 2 kernels:
- Kernel 1 (`triton_poi_fused_div_permute_view_0`): Reads input, multiplies by 0.125, writes to intermediate buffer (3.1M elements)
- Kernel 2 (`triton_poi_fused_as_strided_clone_unsqueeze_1`): Reads from intermediate with overlapping stencil indexing, writes to output (5.9M elements)

The fusion barrier is in `lowering.py:1656`: the `as_strided` lowering calls `x.realize()` on its input, which forces materialization of the upstream pointwise div. This is because as_strided can produce overlapping reads, and fusing a computation into an overlapping-read pattern would require the scheduler to handle multi-use of computed values.

## Kernel Count
- Oracle: 1 kernel
- Inductor: 2 kernels

## Config Exploration
- `combo_kernels = True`: No effect (not a combo kernel pattern)
- `coordinate_descent_tuning = True`: No effect on fusion decisions
- No config can override the as_strided realize barrier

## Design Doc

The as_strided lowering in `lowering.py:1656` unconditionally realizes the input to handle overlapping strides. When the as_strided is followed by a clone (materializing the overlapping view into a fresh contiguous buffer), the entire pattern "pointwise -> as_strided -> clone" could theoretically be fused into one kernel that:
1. For each output element, computes the source index using the stencil pattern
2. Reads from the *original* input (before pointwise)
3. Applies the pointwise op (div by 8)
4. Writes to the contiguous output

This is safe because the clone guarantees no aliasing in the output, and the pointwise is element-independent.

### What enhancement is needed
The scheduler (or a dedicated FX pass) should recognize the pattern:
```
pointwise_producer -> as_strided(overlapping) -> clone(contiguous_format)
```
and fuse it into a single kernel that computes the stencil indexing directly from the original input. The key insight is that clone guarantees the output is non-aliased, so the overlapping nature of the as_strided view is only relevant for *reads*, not writes.

### Affected code
- `/tmp/pytorch-work/torch/_inductor/lowering.py:1656` - as_strided forces realize
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - fusion decisions
- `/tmp/pytorch-work/torch/_inductor/ir.py` - ReinterpretView, realize semantics

### Related repros
- `pointwise_a089fe99095f` (same pattern with constant_pad_nd + as_strided + clone)
- `pointwise_a14dcfc06344` (existing writeup, likely same pattern)
