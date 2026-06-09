# sum_sum_4bd81dea302d Investigation

## Summary
- **Model**: hf_AllenaiLongformerBase_train_005
- **Ratio**: 1.403x (oracle: 62.4us, compile: 87.55us)
- **Classification**: SCATTER_REDUCE

## Root Cause

The oracle computes the complete Longformer overlapping-window return tuple by decoding the generated iota/as_strided duplicate-index scatters into direct source loads, writing both live contiguous [8192, 768] bases used by the returned [768, 8192] transposes, applying the /8 epilogue on the second branch, and accumulating both sibling [768] sums from the same Triton tile.

Inductor materializes two zero-filled [6291456] scatter buffers, runs generic accumulate=True index_puts, reinterprets them through as_strided/view/permute chains, and schedules the reductions and transposed side outputs as separate work.

## Kernel Count
- **Oracle**: 1-2 kernels (structured scatter-reduce with fused transpose and reduction)
- **Inductor**: Multiple kernels (2 scatter fills + 2 index_put + permute/view + 2 reductions)

## Config Exploration
combo_kernels and coordinate_descent_tuning do not help because the fundamental issue is materializing large intermediate scatter buffers (2x 6.3M elements = 48MB) that the oracle avoids entirely.

## Key Files
- `/tmp/pytorch-work/torch/_inductor/fx_passes/` - scatter pattern recognition
- `/tmp/pytorch-work/torch/_inductor/lowering.py` - index_put lowering
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - fusion across scatter/permute

## Design Doc

The fix requires a Longformer indexed scatter-reduce lowering that:
1. Recognizes the overlapping-window iota/as_strided scatter pattern
2. Decodes the structured index relationship (96 heads x 3 overlapping blocks x 512 window x 64 dim -> 96 x 2 x 512 x 64 output)
3. Targets the final live layout directly (transposed [768, 8192])
4. Fuses scale/reduction epilogues (the /8 divisor and [768] sums)
5. Emits the required transposed stores without materializing the 48MB scatter buffers

This is a complex pattern-specific optimization. The structured nature of the overlapping windows (fixed stride relationship between source and dest indices) means a specialized lowering could avoid scatter entirely.
