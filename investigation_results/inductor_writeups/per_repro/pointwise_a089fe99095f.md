# pointwise_a089fe99095f (Longformer Padded Bias Layout)

## Benchmark Result
- Oracle: 9.79 us
- Compiled: 13.15 us
- Ratio: 1.343x
- Status: Regression

## Root Cause

The oracle classifies this as SCHEDULER_FUSION. The repro computes:
1. View f32[2048, 768] as [1024, 2, 768]
2. Add bias f32[768]
3. Reshape to [1024, 2, 12, 64], permute to [2, 12, 1024, 64] -> view as [24, 1024, 64]
4. constant_pad_nd with [0,0,256,256] padding (value -1.0) -> [24, 1536, 64]
5. as_strided with overlapping windows: [24, 4, 768, 64] stride [98304, 16384, 64, 1]
6. Clone to contiguous
7. View to [96, 768, 64], permute [0,2,1] -> [96, 64, 768]

The oracle fuses bias add + constant padding + overlapping stencil materialization + output permute into a single kernel. It computes source coordinates from output indices, checks padding bounds, and either loads from the original mm_46 input (adding bias inline) or writes the pad value -1.0.

Inductor generates 2 kernels:
- Kernel 1 (`triton_poi_fused_add_constant_pad_nd_permute_view_0`): Fuses bias add + constant_pad_nd + permute into intermediate buffer [24, 1536, 64] (2.36M elements)
- Kernel 2 (`triton_poi_fused_as_strided_clone_unsqueeze_1`): Materializes overlapping stencil from intermediate into contiguous output (4.72M elements)

Same root cause as pointwise_9903e70019c2: `lowering.py:1656` forces realization at the as_strided boundary. Inductor fuses everything *up to* the as_strided (bias+pad+permute) into kernel 1, then the as_strided+clone into kernel 2. The oracle avoids the intermediate entirely.

## Kernel Count
- Oracle: 1 kernel (fused add + pad + stencil + permute)
- Inductor: 2 kernels (add+pad+permute | as_strided+clone)

## Config Exploration
- `combo_kernels = True`: No effect
- `coordinate_descent_tuning = True`: No effect on fusion
- No config bypasses the as_strided realize barrier

## Design Doc

This is the same pattern as pointwise_9903e70019c2 but with an additional constant_pad_nd before the as_strided. The oracle handles both the padding logic (bounds check + fill value) and the overlapping stencil in one kernel.

### What enhancement is needed
Same as pointwise_9903e70019c2: recognize `pointwise_chain -> as_strided(overlapping) -> clone(contiguous)` and fuse into one kernel. For the padded variant:
1. Compute source coordinates from output stencil indices
2. Check if source is in padding region -> write pad value
3. Otherwise, read from original input and apply pointwise ops (bias add)

The key safety guarantee is the same: the clone ensures non-aliased output, so overlapping reads are fine.

### Memory traffic savings
- Inductor: reads 2048*768*4=6.3MB (input), writes 24*1536*64*4=9.4MB (intermediate), reads 9.4MB again (stencil), writes 96*768*64*4=18.9MB (output) = ~44MB total traffic
- Oracle: reads 2048*768*4=6.3MB (input), writes 96*64*768*4=18.9MB (output) = ~25MB total traffic
- Savings: eliminates 9.4MB intermediate write + 9.4MB intermediate read = ~19MB saved

### Affected code
- `/tmp/pytorch-work/torch/_inductor/lowering.py:1656` - as_strided forces realize
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - fusion decisions
- `/tmp/pytorch-work/torch/_inductor/ir.py` - ReinterpretView

### Related repros
- `pointwise_9903e70019c2` (same pattern: div + as_strided + clone, Longformer)
- `pointwise_a14dcfc06344` (existing writeup, likely same class)
