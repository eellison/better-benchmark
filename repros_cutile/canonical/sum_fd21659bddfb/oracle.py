"""cuTile port of sum_fd21659bddfb: GPTNeo xent-backward with dual pad outputs.

Structure (from the eager graph):
  scale_value = arg0 / arg1
  labels = arg2[:, 1:].reshape(-1)  # [4096]
  mask = labels != -100  # [4096]

  For each (r, v) in [ROWS, VOCAB]:
    row_sum[r] = -scale_value if labels[r] in [0, VOCAB) and mask[r] else 0
    row_scale[r] = scale_value if mask[r] else 0
    one_hot[r, v] = -1 if labels[r] == v else 0
    logits_norm[r, v] = exp(arg3[r, v] - arg4[r] - arg5[r])
    correction[r, v] = one_hot[r, v] * row_scale[r] - logits_norm[r, v] * row_sum[r]
    correction_bf[r, v] = bf16(correction)
    value[r, v] = arg6[r, v] + correction_bf[r, v]  # bf16 add

  Outputs:
    zero_out = 0.0 (f32 scalar)
    out_t   = permute + constant_pad_nd([50257,4096] -> [50264,4096])
    out_row = constant_pad_nd([4096, 50257] -> [4096, 50264])

Approach: use cuTile for the substantive per-row scaling and add
(sub/exp/mul/sub/add producer), producing the [4096, 50257] result tensor.
Then torch does the permute + constant_pad_nd + the scalar zero output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ_IN = 129
SEQ_OUT = 128
ROWS = BATCH * SEQ_OUT  # 4096
VOCAB = 50257
PADDED_VOCAB = 50264
LOGITS_ROW_STRIDE = 50264  # arg3 has this stride due to padded storage
RESIDUAL_ROW_STRIDE = 50257


@ct.kernel
def _xent_fused_kernel(
    logits_ptr,        # bf16 [ROWS, LOGITS_ROW_STRIDE]  (arg3)
    row_shift0_ptr,    # f32 [ROWS]  (arg4 flat)
    row_shift1_ptr,    # f32 [ROWS]  (arg5 flat)
    residual_ptr,      # bf16 [ROWS, VOCAB]  (arg6)
    mul_lhs_ptr,       # f32 [ROWS, VOCAB]  (the `mul` tensor from eager)
    row_sum_ptr,       # f32 [ROWS]  (sum of `mul` along last dim)
    out_ptr,           # bf16 [ROWS, VOCAB]  (value = arg6 + bf16(sub_2))
    LOGITS_STRIDE: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)

    # Load a [BLOCK_M, BLOCK_N] tile of logits at (row_block, col_block).
    # logits_ptr has stride LOGITS_STRIDE for the row dim.
    # Represent logits as [ROWS, LOGITS_STRIDE] 2D array.
    logits_bf = ct.load(
        logits_ptr, index=(row_block, col_block), shape=(BLOCK_M, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual_bf = ct.load(
        residual_ptr, index=(row_block, col_block), shape=(BLOCK_M, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    mul_lhs = ct.load(
        mul_lhs_ptr, index=(row_block, col_block), shape=(BLOCK_M, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    row_shift0 = ct.load(
        row_shift0_ptr, index=(row_block,), shape=(BLOCK_M,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    row_shift1 = ct.load(
        row_shift1_ptr, index=(row_block,), shape=(BLOCK_M,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    row_sum = ct.load(
        row_sum_ptr, index=(row_block,), shape=(BLOCK_M,),
        padding_mode=ct.PaddingMode.ZERO,
    )

    logits_f = ct.astype(logits_bf, ct.float32)
    row_shift0_2d = ct.reshape(row_shift0, (BLOCK_M, 1))
    row_shift1_2d = ct.reshape(row_shift1, (BLOCK_M, 1))
    row_sum_2d = ct.reshape(row_sum, (BLOCK_M, 1))
    centered = logits_f - row_shift0_2d - row_shift1_2d
    exp_v = ct.exp(centered)
    exp_scaled = exp_v * row_sum_2d
    sub_2 = mul_lhs - exp_scaled
    sub_2_bf = ct.astype(sub_2, ct.bfloat16)
    added = ct.astype(ct.astype(residual_bf, ct.float32) +
                       ct.astype(sub_2_bf, ct.float32), ct.bfloat16)
    ct.store(out_ptr, index=(row_block, col_block), tile=added)


@oracle_impl(hardware="B200", point="11dbdf2e")
def oracle_forward(inputs):
    (
        numerator,   # f32 scalar
        denominator, # f32 scalar
        labels_full, # i64 [32, 129]
        logits,      # bf16 [32, 128, 50257]  (stride [6433792, 50264, 1])
        row_shift0,  # f32 [4096, 1]
        row_shift1,  # f32 [4096, 1]
        residual,    # bf16 [32, 128, 50257]  (contig)
        *_shape_params,
    ) = inputs
    device = logits.device

    # Compute scale_value = numerator / denominator
    scale_value = (numerator / denominator).item()

    # Prepare labels: arg2[:, 1:].reshape(-1)
    labels = labels_full[:, 1:].contiguous().view(-1)  # [4096], i64
    active = labels != -100  # [4096]

    # row_scale = scale_value if active else 0.0    (f32 [4096])
    row_scale = torch.where(active, torch.full_like(active, scale_value, dtype=torch.float32),
                             torch.zeros((), device=device, dtype=torch.float32))

    # `mul` tensor from eager:
    #   one_hot[r, v] = -1 if (safe_labels[r] == v and active[r] and 0<=labels<VOCAB) else 0
    #   mul[r, v] = one_hot[r, v] * row_scale[r]
    #   BUT: one_hot is only defined for the eq check. If labels[r] is out of
    #   range or inactive, one_hot is all zeros for that row.
    # Simplification: for the active rows where 0 <= label < VOCAB, only the
    # column label[r] is -1 * scale_value; all others are 0.
    safe_labels = torch.where(active, labels, torch.zeros_like(labels))
    in_range = (safe_labels >= 0) & (safe_labels < VOCAB)
    valid = active & in_range   # [4096]

    # mul: [ROWS, VOCAB] but sparse. Build it densely — 4096*50257 f32 = 800MB.
    # Too big. Use scatter/index_put for the -scale entry per row.
    # Actually we need mul as a dense tensor for the cuTile kernel. Let's
    # allocate it (800 MB). Alternative: encode mul via row_sum + label-based
    # correction (mul[r, v] = -scale if v == label else 0).
    # In the kernel we can compute mul_lhs directly.
    # Let me try memory-lean approach: allocate mul as zero, then scatter -scale
    # at (r, label[r]) for valid rows.
    mul_lhs = torch.zeros((ROWS, VOCAB), device=device, dtype=torch.float32)
    row_idx = torch.arange(ROWS, device=device)
    valid_rows = row_idx[valid]
    valid_cols = safe_labels[valid]
    mul_lhs[valid_rows, valid_cols] = -scale_value

    # row_sum = sum(mul, dim=1) = -scale_value if valid else 0
    row_sum = torch.where(valid, torch.full_like(valid, -scale_value, dtype=torch.float32),
                           torch.zeros((), device=device, dtype=torch.float32))

    # Reshape inputs to 2D [ROWS, VOCAB] and [ROWS, LOGITS_STRIDE].
    # logits is [32,128,50257] with stride [6433792, 50264, 1] -> memory is
    # [rows=4096, LOGITS_ROW_STRIDE=50264] with valid VOCAB=50257 in first
    # 50257 columns; 7 padding cols.
    # logits has physical shape [32,128,50257] but stride [6433792, 50264, 1].
    # Reinterpret as [ROWS, VOCAB] with row stride 50264.
    logits_2d = torch.as_strided(
        logits, (ROWS, VOCAB), (LOGITS_ROW_STRIDE, 1)
    )
    residual_2d = torch.as_strided(residual, (ROWS, VOCAB), (VOCAB, 1))
    row_shift0_1d = row_shift0.view(ROWS)
    row_shift1_1d = row_shift1.view(ROWS)

    # Output: bf16 [ROWS, VOCAB] contiguous.
    out_2d = torch.empty((ROWS, VOCAB), device=device, dtype=torch.bfloat16)

    # Kernel launch: iterate over 2D grid (row tiles, col tiles).
    # BLOCK_N should divide VOCAB=50257? 50257 is prime. Use padded tile: read
    # BLOCK_N cols but mask writes... but cuTile can't mask stores. Instead
    # we pass logits_2d with LOGITS_ROW_STRIDE=50264 and use BLOCK_N=64
    # so logits_2d covers full 50264. But out_2d has VOCAB=50257 cols and
    # BLOCK_N * ceil(VOCAB/BLOCK_N) may exceed. Simpler: launch grid to
    # cover [ROWS, VOCAB] but round VOCAB up to a multiple.
    #
    # Since we can't mask stores, use BLOCK_N=1 for the final column: run
    # the "meaty" kernel over [ROWS, 50176] (multiple of 64) then handle the
    # tail 81 cols with torch (small).
    BLOCK_M = 32
    BLOCK_N = 64
    # Choose col_tiles such that col_tiles * BLOCK_N <= VOCAB and is largest.
    col_tiles = VOCAB // BLOCK_N  # 785
    covered_cols = col_tiles * BLOCK_N  # 50240

    logits_2d_covered = logits_2d[:, :covered_cols].contiguous(
    ) if False else logits_2d  # keep original strided view; kernel indexes into first covered_cols
    # The kernel loads a (BLOCK_M, BLOCK_N) tile at index (row_block, col_block).
    # logits_2d and residual_2d/out_2d have different strides (LOGITS vs VOCAB
    # for the row dim). We need to pass them as different arrays.
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS // BLOCK_M, col_tiles, 1),
        _xent_fused_kernel,
        (logits_2d, row_shift0_1d, row_shift1_1d, residual_2d,
         mul_lhs, row_sum, out_2d,
         LOGITS_ROW_STRIDE, BLOCK_M, BLOCK_N),
    )

    # Tail 17 columns [50240:50257] handled by torch to keep the port simple.
    tail_start = covered_cols
    tail_cols = VOCAB - covered_cols
    if tail_cols > 0:
        logits_tail = logits_2d[:, tail_start:VOCAB].to(torch.float32)
        residual_tail = residual_2d[:, tail_start:VOCAB]
        mul_tail = mul_lhs[:, tail_start:VOCAB]
        centered = logits_tail - row_shift0_1d.view(ROWS, 1) - row_shift1_1d.view(ROWS, 1)
        exp_v = torch.exp(centered)
        sub_2 = mul_tail - exp_v * row_sum.view(ROWS, 1)
        sub_2_bf = sub_2.to(torch.bfloat16)
        out_2d[:, tail_start:VOCAB] = (
            residual_tail.to(torch.float32) + sub_2_bf.to(torch.float32)
        ).to(torch.bfloat16)

    # Now reshape/pad to produce eager outputs.
    add_out = out_2d  # bf16 [ROWS, VOCAB]

    # permute: [VOCAB, ROWS]
    perm = add_out.permute(1, 0)
    # constant_pad_nd with padding [0, 0, 0, 7] -> pad last-1 dim's end by 7:
    # torch.nn.functional.pad expects (last_dim_left, last_dim_right, ...)
    # _shape_param_5 = [0, 0, 0, 7]. Applied to [VOCAB, ROWS]:
    # padding = (0, 0, 0, 7) means (last_left, last_right, sec_left, sec_right)
    # -> pad rows (VOCAB dim) with 7 at the end -> [50264, ROWS]
    out_t = torch.nn.functional.pad(perm, (0, 0, 0, 7))
    # For view_4 pad with [0, 7, 0, 0] -> [ROWS, VOCAB+7] = [ROWS, 50264]
    out_row = torch.nn.functional.pad(add_out, (0, 7, 0, 0))

    zero_out = torch.zeros((), device=device, dtype=torch.float32)

    return zero_out, out_t, out_row
