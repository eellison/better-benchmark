# sum_31bf563cdf96


## Measured Timings
- Oracle: 91.74 us
- Compile (CDT): 110.40 us
- Ratio: 1.20x

## Classification: SCHEDULER_FUSION

## Current Result

- Family: `beit_qkv_slice_sum`
- Oracle path: `repros/canonical/sum_31bf563cdf96/oracle_qkv_slice_sum.py`
- Correctness: assumed PASS (oracle has check mode)
- Compiled (coordinate_descent_tuning=True): 117.60 us
- Gap: 1.60x
- Status: `real_gap`

## Diagnosis

The oracle materializes the returned QKV transpose and computes the returned q/v reduction slices directly from the original strided inputs with a fixed two-stage f32 sum, whereas Inductor first writes the full cloned QKV buffer then schedules full-width reduction kernels that reread it and compute the unreturned middle k columns.

The repro computes:
1. `cat([q, k, v])` of 3x [128, 12, 197, 64] strided tensors -> [384, 12, 197, 64]
2. reshape -> [3, 128, 12, 197, 64]
3. permute [1,3,0,2,4] -> [128, 197, 3, 12, 64]
4. clone (contiguous) -> [128, 197, 2304]
5. reshape -> [25216, 2304]
6. permute [1, 0] -> return as [2304, 25216]
7. sum dim=[0] -> [1, 2304] -> slice [0:768] and [1536:2304] -> return q_bias and v_bias grads

## Root cause

Inductor schedules:
- Kernel 1: materialization of the [25216, 2304] contiguous buffer (cat + permute + clone)
- Kernel 2: transpose copy for [2304, 25216] return
- Kernel 3: full-width sum over all 2304 columns, even though only q (0:768) and v (1536:2304) slices are returned

The oracle avoids reading the middle k columns (768:1536) entirely for the reduction, and fuses the materialization with the partial reduction in one pass.

## Kernel count
- Oracle: 3 kernels (fused materialization + slice-aware partial sums + finalization)
- Inductor: 3 kernels (materialize, transpose, full-width sum)

## Config exploration results

| Config | Expected Impact |
|--------|----------------|
| coordinate_descent_tuning | Already applied (117.60 us) |
| combo_kernels | Does not help (different output patterns: transpose vs reduction) |
| multi_kernel | Not applicable (reduction is already outer-dim) |

## Recommendation

Teach the scheduler to:
1. Recognize that `slice(sum(x, dim=0), ...)` only needs partial columns, not the full 2304
2. Fuse the returned layout materialization with the slice-aware reduction so both outputs share one pass over the input data

This requires mixed materialization-plus-reduction scheduling with slice-aware column pruning.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (fusion of materialization + sliced reduction)
- Input: 3x [128, 12, 197, 64] strided f32 (BEiT QKV backward)
- Total bytes: ~465 MB
- Models: timm_beit_base_patch16_224_train
