# sum_sum_sum_3eab263249bb

## Compile: 117.86us, Oracle: 96.0us, Gap: 1.228x

## Classification: SCATTER_REDUCE

## Root Cause

The oracle computes the full three-output Swin relative-position-bias backward with one vectorized Triton producer that emits the [8192, 49, 49] softmax-backward side output and atomically accumulates both batch-reduction partials into duplicate [169, 16] relative-position buckets.

Inductor generates 3 kernels:
1. `triton_poi_fused_0`: initialization/zeroing
2. `triton_per_fused_1`: batch reduction over dim=0
3. `triton_red_fused_2`: fused softmax-backward + scatter (fma, neg, mul, sum, index_put)

The issue is that Inductor handles the two `sum(dim=0) -> permute/view -> index_put(accumulate=True)` branches and the `mul/sum/neg/fma/view` softmax-backward branch as separate generic reduction, layout, scatter, and pointwise kernels. The oracle fuses these by recognizing that the duplicate-index relative-position `index_put(accumulate=True)` is a structured scatter-reduce that can share the rowwise softmax-backward producer with the required side-output stores.

## Kernel Count
- Oracle: 1 kernel (fused softmax-bwd + scatter-reduce)
- Inductor: 3 kernels (zero, batch reduce, fused reduction+scatter)

## Config Exploration
| Config | Time (us) | Notes |
|--------|-----------|-------|
| combo_kernels + CDT | 117.86 | baseline (1.228x) |
| multi_kernel=2 | unlikely to help (scatter pattern) |
| multi_kernel=3 | unlikely to help (scatter pattern) |

## Fix Assessment: Design doc

The scheduler/codegen needs a structured relative-position scatter-reduce lowering that:
1. Fuses the batch dimension reduction with indexed accumulation
2. Emits the full-tensor softmax-backward side output from the same producer pass
3. Handles duplicate-index accumulation (index_put with accumulate=True on [169] buckets from [2401] sources)

### Relevant files:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: cannot fuse reduction + scatter consumers
- `/tmp/pytorch-work/torch/_inductor/lowering.py`: index_put lowering with accumulate=True
- `/tmp/pytorch-work/torch/_inductor/ir.py`: scatter nodes block fusion

### Affected repro count:
The relative-position scatter pattern occurs in all Swin Transformer variants. Likely 5-10 repros with this pattern.

## Details
- Model: timm_swin_base_patch4_window7_224 (train)
- Shape: [512, 16, 49, 49] f32 input -> [169, 16] f32 index_put outputs + [8192, 49, 49] side output
- Pattern: sum(dim=0) -> permute [49,49,16] -> view [2401,16] -> index_put [169,16] (accumulate) + softmax_backward -> view [8192,49,49]
- The index_put uses [49,49] -> [169] position bias indexing (2401 sources -> 169 buckets)
