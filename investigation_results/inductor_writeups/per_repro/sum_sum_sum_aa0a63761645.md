# sum_sum_sum_aa0a63761645

## Classification: COOPERATIVE_SPLIT_K

## Oracle: oracle_cooperative_split_k.py

## Model: torchbench_phlippe_resnet_train_001

## Measurements

- Compiled (default + combo + cd): 21.2 us
- Compiled (multi_kernel=1): 38.0 us
- Compiled (multi_kernel=2): 39.4 us
- Compiled (multi_kernel=3): 37.3 us
- Oracle: 14.4 us
- Ratio: 1.474x
- Oracle correctness: PASS

## Diagnosis

The oracle computes the complete ResNet BN-backward-style return tuple by sharing one split-K Triton producer for the masked add, `sum(where)`, and `sum(where * (arg66_1 - arg110_1))`, finalizing the two returned [32] channel vectors, and writing the dependent contiguous `[128, 32, 16, 16]` epilogue with the target strides.

Inductor schedules the masked add producer, channel reductions, vector epilogues, and full-tensor BN-backward epilogue as separate generic pointwise/reduction kernels over materialized intermediates.

## Config exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (combo + cd) | 21.2 | 1.474x |
| multi_kernel=1 | 38.0 | 2.639x |
| multi_kernel=2 | 39.4 | 2.736x |
| multi_kernel=3 | 37.3 | 2.590x |

The default combo+cd config is the best (1.474x). Multi_kernel actually makes performance worse by 2x, suggesting combo_kernels is doing important fusion that multi_kernel disrupts.

## Root cause

Inductor's scheduler/codegen has no cooperative split-K multi-output reduction template that:
1. Finalizes shared channel partials from the spatial domain [128, 32, 16, 16] -> [32]
2. Feeds both sibling vector outputs ([32] channel sums)
3. Fuses a dependent full-tensor side store ([128, 32, 16, 16] BN-backward epilogue)

The oracle reads the `[128, 32, 16, 16]` input once, computes all channel statistics with split-K parallelism, and writes all outputs without materializing intermediates.

## Fix path

- Add a channel-reduction split-K template in `torch/_inductor/choices.py`
- Template must accumulate multiple summaries per channel, finalize them once, and fuse vector + full-tensor epilogues
- File: `torch/_inductor/scheduler.py` (multi-output channel reduction fusion)
- File: `torch/_inductor/codegen/triton.py` (split-K finalization + epilogue codegen)

## Status: needs_work (1.47x gap, multi_kernel makes it worse)
