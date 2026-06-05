# amax_sum_4bd27b112605

## Compile: 100.1us, Oracle: 66.9us, Gap: 1.50x

## Diagnosis: NEW_PATTERN

## Root cause: Inductor lowers the T5 bidirectional relative-position attention softmax as a looped 2-pass reduction kernel that recomputes the log-bucket index calculation (iota/abs/log/minimum/where/embedding) in both passes plus writes/rereads the intermediate scores tensor between passes. The oracle computes the same bidirectional bucket selection and bias gather in a single persistent Triton kernel with block_k=2048, keeping the full row in registers and doing softmax in one pass without intermediate materialization.

The specific inefficiencies are:
1. Two-pass looped reduction: pass 1 accumulates online max/sum and writes scores to `in_out_ptr0`; pass 2 rereads those scores to normalize. Oracle does this in one pass.
2. Redundant bucket recomputation: The relative-position bucket logic (abs, log, threshold comparisons, direction offset) is recomputed for every element in the first pass but the result isn't reused in the second pass (scores are re-read from memory instead).
3. device_assert overhead: `tl.device_assert` for the embedding index bounds check adds branch overhead in the inner loop.
4. Tautological zero mask: The structured mask `where(True, 0.0, -65504.0)` always selects 0.0, adding dead `tl.where` + `tl.full` instructions.

## Kernel count: Inductor 1, Oracle 1

## Config exploration:
- `coordinate_descent_tuning=True`: 100.1us (baseline)
- `combo_kernels=True`: no change (already 1 kernel)
- `triton.multi_kernel=3`: 100.1us (no help)
- `max_autotune=True`: 100.1us (no help)
- `assert_indirect_indexing=False`: 103.5us (marginal, within noise)
- `TORCHINDUCTOR_FORCE_PERSISTENT_REDUCTIONS=1`: 111.6us (worse - Inductor's persistent mode doesn't help because it still materializes the intermediate)

## Fix path: This requires a NEW_PATTERN lowering that:
1. Recognizes the T5 bidirectional relative-position bucket computation as a cheap recomputable producer
2. Eliminates the tautological zero mask (constant True predicate -> dead code)
3. Uses a persistent softmax template that keeps the full 2048-element row in registers
4. Recomputes the bucket/bias inline in the single-pass softmax instead of materializing

Key files:
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` - pattern registration
- `/tmp/pytorch-work/torch/_inductor/choices.py` - persistent reduction threshold (2048 rows may not qualify)
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - fusion decisions for pointwise+reduction

## Status: design_todo

## Details

- Model: hf_T5Large (train, bidirectional attention)
- Pattern: iota -> abs -> log buckets -> embedding gather -> add BMM scores -> softmax -> fp16 cast -> expand -> view
- Shape: [8, 2048, 2048] fp16 (8 heads, 2048 seq len)
- The oracle pre-computes bucket thresholds as integer comparisons (avoiding log/float arithmetic in the inner loop)
- Related to amax_sum_amax_2a81770def44 (same T5 pattern but with dropout epilogue, dual outputs)
- Row length 2048 is borderline for persistent vs looped; oracle proves persistent is viable here
