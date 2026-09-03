"""cuTile port of sum_sum_5618630d0a84: XLNet large-vocab loss backward.

Materializes the bf16 [8192, 32000] loss-backward tensor via one cuTile
kernel that also emits per-row-block column partials (mirroring Triton's
_materialize_partials_kernel), followed by a finalize kernel that reduces
partials into the final column sum with a bf16 round trip.

Native cuTile fp32/bf16 arithmetic is round-to-nearest, matching Triton's
inline PTX `add.rn`, `mul.rn`, `div.rn` intrinsics and
`fp_downcast_rounding="rtne"`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
VOCAB = 32000


@ct.kernel
def _materialize_partials_kernel(
    numerator_ptr,   # f32 scalar
    denominator_ptr, # f32 scalar
    labels_ptr,      # i64 [ROWS]
    logits_ptr,      # bf16 [ROWS, VOCAB]
    shift0_ptr,      # f32 [ROWS]
    shift1_ptr,      # f32 [ROWS]
    residual_ptr,    # bf16 [ROWS, VOCAB]
    out_ptr,         # bf16 [ROWS, VOCAB]
    partial_ptr,     # f32 [num_row_blocks, VOCAB]
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)

    num_tile = ct.load(numerator_ptr, index=(0,), shape=(1,))
    den_tile = ct.load(denominator_ptr, index=(0,), shape=(1,))
    scale = num_tile / den_tile  # scalar f32

    # Load labels (rows,).
    labels = ct.load(labels_ptr, index=(row_block,), shape=(BLOCK_R,))
    valid = labels != -100
    target = ct.where(valid, labels, 0)
    # scale is a tile of shape (1,); broadcast: row_grad shape (BLOCK_R,)
    scale_bcast = ct.broadcast_to(scale, (BLOCK_R,))
    neg_scale = -scale_bcast
    # neg_scale.to(bf16).to(f32) — bf16 roundtrip
    row_grad = ct.astype(ct.astype(neg_scale, ct.bfloat16), ct.float32)
    row_grad = ct.where(valid, row_grad, 0.0)

    logits = ct.load(logits_ptr, index=(row_block, col_block), shape=(BLOCK_R, BLOCK_C))
    logits_f = ct.astype(logits, ct.float32)
    shift0 = ct.load(shift0_ptr, index=(row_block,), shape=(BLOCK_R,))
    shift1 = ct.load(shift1_ptr, index=(row_block,), shape=(BLOCK_R,))
    shift0_2d = ct.reshape(shift0, (BLOCK_R, 1))
    shift1_2d = ct.reshape(shift1, (BLOCK_R, 1))
    shifted = logits_f - shift0_2d - shift1_2d
    # bf16 rounding
    shifted_bf16 = ct.astype(shifted, ct.bfloat16)
    shifted_f = ct.astype(shifted_bf16, ct.float32)
    exp_val = ct.exp(shifted_f)

    # sparse: [BLOCK_R, BLOCK_C] where col == target[row]
    cols = col_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    target_2d = ct.reshape(target, (BLOCK_R, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_C))
    valid_2d = ct.reshape(valid, (BLOCK_R, 1))
    match = (cols_2d == target_2d) & valid_2d
    row_grad_2d = ct.reshape(row_grad, (BLOCK_R, 1))
    sparse = ct.where(match, row_grad_2d, 0.0)

    dense_delta = sparse - exp_val * row_grad_2d
    delta_bf16 = ct.astype(dense_delta, ct.bfloat16)
    residual = ct.load(residual_ptr, index=(row_block, col_block), shape=(BLOCK_R, BLOCK_C))
    residual_f = ct.astype(residual, ct.float32)
    delta_f = ct.astype(delta_bf16, ct.float32)
    out_bf16 = ct.astype(residual_f + delta_f, ct.bfloat16)

    ct.store(out_ptr, index=(row_block, col_block), tile=out_bf16)

    # Partial column sum for this row block
    out_f = ct.astype(out_bf16, ct.float32)
    partial = ct.sum(out_f, axis=0)  # (BLOCK_C,)
    ct.store(partial_ptr, index=(row_block, col_block),
             tile=ct.reshape(partial, (1, BLOCK_C)))


@ct.kernel
def _finalize_sum_kernel(
    partial_ptr,     # f32 [num_row_blocks, VOCAB]
    sum_out_ptr,     # f32 [VOCAB]
    NUM_ROW_BLOCKS: ct.Constant[int],
    BLOCK_RB: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    values = ct.load(partial_ptr, index=(0, col_block),
                     shape=(BLOCK_RB, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    row_idx = ct.arange(BLOCK_RB, dtype=ct.int32)
    row_mask = ct.reshape(row_idx < NUM_ROW_BLOCKS, (BLOCK_RB, 1))
    valid = ct.where(row_mask, values, 0.0)
    reduced = ct.sum(valid, axis=0)
    # bf16 round trip
    reduced_bf = ct.astype(reduced, ct.bfloat16)
    reduced_f = ct.astype(reduced_bf, ct.float32)
    ct.store(sum_out_ptr, index=(col_block,), tile=reduced_f)


def _next_p2(v):
    return 1 << (int(v) - 1).bit_length()


# hf XLNetLMHeadModel train, labels [8192], dense vocab [8192, 32000].
@oracle_impl(
    hardware="B200",
    point="011490da",
    BLOCK_R=64,
    BLOCK_C=32,
)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int):
    (
        arg0_1,  # numerator f32 scalar
        arg1_1,  # denominator f32 scalar
        arg2_1,  # labels i64 [16, 512]
        arg3_1,  # logits bf16 [16, 512, 32000]
        arg4_1,  # shift0 f32 [8192, 1]
        arg5_1,  # shift1 f32 [8192, 1]
        arg6_1,  # residual bf16 [16, 512, 32000]
        *_shape_params,
    ) = inputs

    if VOCAB % BLOCK_C != 0:
        raise NotImplementedError(f"BLOCK_C={BLOCK_C} does not divide VOCAB={VOCAB}")
    if ROWS % BLOCK_R != 0:
        raise NotImplementedError(f"BLOCK_R={BLOCK_R} does not divide ROWS={ROWS}")

    device = arg3_1.device
    out = torch.empty_strided(
        (ROWS, VOCAB), (VOCAB, 1), device=device, dtype=torch.bfloat16
    )
    num_row_blocks = ROWS // BLOCK_R
    partial = torch.empty(
        (num_row_blocks, VOCAB), device=device, dtype=torch.float32
    )
    sum_out = torch.empty((VOCAB,), device=device, dtype=torch.float32)

    # Reshape / view helpers.
    num_scalar = arg0_1.view(1)
    den_scalar = arg1_1.view(1)
    labels_1d = arg2_1.view(-1)
    logits_2d = arg3_1.view(ROWS, VOCAB)
    shift0_1d = arg4_1.view(-1)
    shift1_1d = arg5_1.view(-1)
    resid_2d = arg6_1.view(ROWS, VOCAB)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_row_blocks, VOCAB // BLOCK_C, 1),
        _materialize_partials_kernel,
        (
            num_scalar, den_scalar, labels_1d, logits_2d,
            shift0_1d, shift1_1d, resid_2d, out, partial,
            BLOCK_R, BLOCK_C,
        ),
    )
    ct.launch(
        stream,
        (VOCAB // BLOCK_C, 1, 1),
        _finalize_sum_kernel,
        (partial, sum_out, num_row_blocks, _next_p2(num_row_blocks), BLOCK_C),
    )

    return out, torch.as_strided(out, (VOCAB, ROWS), (1, VOCAB)), sum_out
