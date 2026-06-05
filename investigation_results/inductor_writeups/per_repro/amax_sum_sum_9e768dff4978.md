# amax_sum_sum_9e768dff4978

## Classification: `NEW_PATTERN`

## Pattern

Shifted-label ignore-index cross-entropy mean (online softmax variant):
constant_pad_nd/slice/clone/view + bf16-to-f32 + amax/sub/exp/sum/log/sub + gather + ne/where/sum/div

- Model: vllm_Qwen_Qwen3-0.6B (inference/training)
- Shape: logits bf16[2048, 151936] (very large vocab), tokens i64[4, 512]
- Reduction dim (rnumel): 151936 (extremely large vocab)
- Output: scalar f32 (mean cross-entropy loss)

## Measurements

| Metric | Value |
|--------|-------|
| Oracle | 171.84 us |
| Compile (best, CD tuning) | 266.11 us |
| Ratio | 1.549x |
| Kernel count (Inductor) | 2 |
| Kernel count (Oracle) | 2 |
| Status | GOOD |

## Diagnosis

The `cross_entropy_loss_online` fusion pass **does fire** for this repro (confirmed
in output_code comments). Inductor successfully fuses the online softmax + target
logit gather into one row-reduction kernel, then reduces per-row losses to a scalar
in a second kernel. The oracle uses an identical 2-kernel structure.

The 1.55x gap stems from **three compounding inefficiencies** in the Inductor codegen
compared to the oracle:

### 1. Shifted-label overhead in the second kernel

The oracle shifts labels inline in the first (row) kernel: it loads `labels[row+1]`
directly and checks validity, storing per-row loss and valid-count. Inductor's second
kernel (the scalar mean) must re-derive the shifted label by computing the
constant_pad_nd/slice/clone/view chain inline, producing extra arithmetic and
conditional loads (boundary checks for `pos < seq_len - 1`).

### 2. `device_assert` in the second kernel

Inductor emits `tl.device_assert(((0 <= tmp15) & (tmp15 < 151936)) | ~mask)` for the
indirect indexing into the logits tensor when computing the target logit in the second
kernel. The oracle does not emit bounds checks since it validates shapes upfront.

### 3. Block size / tiling for rnumel=151936

The oracle uses `block_n=8192, num_warps=16` for the very large vocab, hand-tuned to
maximize occupancy on this specific shape. Inductor's coordinate descent tuning may
converge to a slightly less optimal configuration for this unusual vocab size.

### Root cause summary

The online cross-entropy fusion fires correctly, but the **shifted-label preprocessing
is not fused into the row kernel**. Instead, label shifting is deferred to the scalar
reduction kernel, forcing extra computation and memory accesses in what should be a
trivial reduce. The oracle shows that computing the shifted label + validity mask in
the row kernel (alongside the online softmax) and emitting per-row loss+valid is the
optimal structure.

## Config Exploration

| Config | Time (us) | Notes |
|--------|-----------|-------|
| CD tuning + combo_kernels (default) | 266 | baseline |
| + assert_indirect_indexing=False | ~266 | no measurable change (assert in 2nd kernel only) |
| + scalar_reduction_accumulators | ~266 | already enabled effectively |
| + multi_kernel=3 | ~601 | much worse (launch overhead) |

## File/Line References

- Online cross-entropy fusion pass: `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py:295` (`_cross_entropy_gather_fusion_pass`)
- `cross_entropy_loss_online` prim: `/tmp/pytorch-work/torch/_inductor/inductor_prims.py`
- Label shift not fused into row kernel: the `constant_pad_nd` + `slice` + `clone` + `view` + `ne` + `where` ops remain in the second kernel's reduction loop

## Inductor Closure Path

**Design doc**: The fix requires enhancing the `_cross_entropy_gather_fusion_pass` to
recognize the shifted-label pattern (constant_pad_nd + slice producing the target
indices) and fold the label shift into the fused online softmax row kernel. This would:

1. Compute `shifted_label = labels[batch, pos+1]` with boundary check inside the row
   kernel
2. Store per-row `(loss, is_valid)` from kernel 1
3. Keep kernel 2 as a trivial scalar mean over `(loss, valid_count)` with no label logic

This is the same pattern as `amax_sum_sum_846668f0b88f` (GPT-J shifted cross-entropy)
but at a much larger vocab (151936 vs 50400), making the gap more pronounced due to
the larger intermediate that would be avoided.

## Related Repros

- `amax_sum_sum_846668f0b88f`: same shifted-label cross-entropy pattern, smaller vocab (50400), 1.66x gap
- `amax_sum_sum_4c98004c3aa3`: non-shifted cross-entropy, rnumel=29056, 1.33x gap
- `amax_sum_sum_6939e2db29e3`: same as above, training variant
