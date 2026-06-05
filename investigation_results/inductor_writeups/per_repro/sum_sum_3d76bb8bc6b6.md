# sum_sum_3d76bb8bc6b6

## Compile: 47us, Oracle: 29.5us, Gap: 1.59x

## Diagnosis: COOPERATIVE_SPLIT_K

## Root cause

Inductor schedules the DenseNet batch-norm-backward sibling channel reductions (over N=128, H=32, W=32 = 131072 elements per channel, 48 channels), dependent broadcast arithmetic, and sliced add output as 3 separate kernels:
1. Split reduction: 48*25=1200 CTAs (partial sums)
2. Finalization: 48 CTAs (sum partials)
3. Pointwise epilogue: ~49K CTAs (re-reads ALL original inputs + reduction results)

The oracle uses just 2 kernels:
1. Split reduction with atomics: 768 CTAs (8 channel tiles * 128 spatial tiles)
2. Epilogue: 65536 CTAs, but only processes OUT_SLICE_NUMEL=2M elements (16/48 channels)

The oracle's key advantages:
1. Atomic accumulation (no finalization kernel needed)
2. Dead channel elimination: only 16/48 output channels are used, saving 2/3 of epilogue bandwidth

## Config exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (split, CDT) | 47.0 | 1.59x |
| cooperative_reductions=True | 42.7 | 1.45x |
| force_cooperative=True | 45.4 | 1.54x |
| split_reductions=False | 58.5 | 1.98x |

## Status: design_todo (remaining gap from dead channel elimination)

The gap could be further closed by:
1. **Dead output slice propagation**: Push the `slice_tensor_1 = add_tensor[:, 0:16]` backward through the pointwise epilogue so only 16/48 channels are computed. This would save ~67% of epilogue bandwidth.
2. **Cooperative reduction** (when enabled): fuses the reduction with the epilogue in one kernel, saving one full re-read of inputs.

## Fix path

- **Immediate**: Enable cooperative_reductions for this case -> 1.45x (from 1.59x)
- **Full fix**: Dead output slice propagation FX pass that recognizes when a slice immediately follows a pointwise and rewrites the pointwise to only compute the needed elements.

## Files
- `/tmp/pytorch-work/torch/_inductor/choices.py`: cooperative threshold (xhint=48 <= 64, qualifies)
- `/tmp/pytorch-work/torch/_inductor/fx_passes/`: needs new slice-backward propagation pass
