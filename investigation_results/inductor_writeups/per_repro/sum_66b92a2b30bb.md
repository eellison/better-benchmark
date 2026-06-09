# sum_66b92a2b30bb

## Compile: 17.18us, Oracle: 13.25us, Gap: 1.297x

## Classification: PERMUTE_SIDE_SIBLING

## Root Cause

The oracle streams through the contiguous [8, 6, 1500, 64] (stride [576000, 64, 384, 1]) input once with a scale factor of 0.125, writing both the [384, 12000] transposed permuted output AND accumulating the [384] column sum atomically in a single kernel pass using tiled BLOCK_M=256, BLOCK_N=32 over the flattened [12000, 384] view.

Inductor generates 3 kernels:
1. `triton_poi_fused_mul_permute_view_0`: materializes the scaled+permuted tensor
2. `triton_red_fused_mul_permute_sum_view_1`: partial column reduction
3. `triton_red_fused_mul_permute_sum_view_2`: final column reduction

The fundamental issue is identical to sum_13195092a57b: Inductor cannot fuse a layout-changing materialization (permute side output) with a sibling reduction (column sum) from the same producer.

## Kernel Count
- Oracle: 1 kernel (scale + store transposed + atomic column sum)
- Inductor: 3 kernels (scale+permute store, partial reduce, final reduce)

## Config Exploration
| Config | Time (us) | Notes |
|--------|-----------|-------|
| combo_kernels + CDT | 17.18 | baseline (1.297x) |
| multi_kernel=2 | expected worse (same as sum_13195092a57b) |
| multi_kernel=3 | expected worse |

Config changes do not help -- this is a structural scheduler limitation.

## Fix Assessment: Design doc (same root cause as sum_13195092a57b)

Identical fix needed as sum_13195092a57b. The scheduler needs a multi-output template that combines layout-changing materialization with sibling atomic reduction in one kernel.

### Additional complexity for this repro:
The input has stride [576000, 64, 384, 1] (not contiguous), so the kernel must handle the permute [0,2,1,3] -> view [12000, 384] layout transformation as well as the 0.125 scale factor. The oracle handles this by treating the input's physical layout directly.

### Relevant files:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: fusion decisions
- `/tmp/pytorch-work/torch/_inductor/ir.py`: realize_hint blocks fusion for layout changes

### Affected repro count:
Same family: sum_13195092a57b (1.224x), sum_43bbf21a2e59 (1.065x), sum_7953a8b0bbba (1.504x), sum_b8db5e701976.

## Details
- Model: torchbench_hf_Whisper (train)
- Shape: input [8, 6, 1500, 64] stride=(576000, 64, 384, 1) f32
- Pattern: permute [0,2,1,3] -> view [8,1500,384] -> view [12000,384] -> mul(0.125) -> permute [384,12000] (output 0) + sum(dim=0) [384] (output 1)
