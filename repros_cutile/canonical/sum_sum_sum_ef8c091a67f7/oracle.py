"""cuTile port of sum_sum_sum_ef8c091a67f7: XLNet LayerNorm backward.

Structure mirrors Triton's 2-kernel plan:
  1) _row_partials_kernel: per row group, compute row-local LN-backward
     (mul_6 grad, dropped bf16 output) with row_sum/row_dot reductions,
     plus column partials for sum_3 (clone*arg3), sum_4 (clone), sum_5
     (bf16 rounded).
  2) _finalize_partials_kernel: sum column partials across groups to
     produce sum_3, sum_4, sum_5.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SEQ = 512
SEGMENTS = 16
ROWS = SEQ * SEGMENTS  # 8192
HIDDEN = 1024
DROP_SCALE = 1.1111111111111112


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


def _shape(shape):
    return tuple(int(d) for d in shape)


@ct.kernel
def _row_partials_kernel(
    source_ptr,      # bf16 [SEGMENTS, SEQ, HIDDEN]  (permuted src)
    keep0_ptr,       # f32 [ROWS, HIDDEN]
    weight_ptr,      # f32 [HIDDEN]
    rhs_ptr,         # f32 [ROWS, HIDDEN]
    scale_ptr,       # f32 [ROWS]
    keep1_ptr,       # f32 [ROWS, HIDDEN]
    grad_out_ptr,    # f32 [ROWS, HIDDEN]
    bf16_out_ptr,    # bf16 [ROWS, HIDDEN]
    partial_x_normed_ptr, # f32 [num_groups, HIDDEN]
    partial_x_ptr,        # f32 [num_groups, HIDDEN]
    partial_bf16_ptr,     # f32 [num_groups, HIDDEN]
    ROWS_PER_GROUP: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    ROWS_: ct.Constant[int],
    SEGMENTS_: ct.Constant[int],
    DROP_SCALE_: ct.Constant[float],
):
    group = ct.bid(0)

    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(HIDDEN_,)), ct.float32)
    weight_1d = weight

    acc_clone_rhs = ct.zeros((HIDDEN_,), dtype=ct.float32)
    acc_clone = ct.zeros((HIDDEN_,), dtype=ct.float32)
    acc_bf16 = ct.zeros((HIDDEN_,), dtype=ct.float32)

    for row_offset in ct.static_iter(range(ROWS_PER_GROUP)):
        row = group * ROWS_PER_GROUP + row_offset

        # source is at [segment, seq, :] where seq = row // SEGMENTS, segment = row % SEGMENTS.
        seq = row // SEGMENTS_
        segment = row - seq * SEGMENTS_

        source_bf = ct.load(source_ptr, index=(segment, seq, 0),
                            shape=(1, 1, HIDDEN_),
                            padding_mode=ct.PaddingMode.ZERO)
        source_f = ct.astype(source_bf, ct.float32)
        source_1d = ct.reshape(source_f, (HIDDEN_,))

        keep0 = ct.astype(
            ct.load(keep0_ptr, index=(row, 0), shape=(1, HIDDEN_),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        keep0_1d = ct.reshape(keep0, (HIDDEN_,))
        keep0_scaled = keep0_1d * DROP_SCALE_
        clone = source_1d * keep0_scaled

        rhs = ct.load(rhs_ptr, index=(row, 0), shape=(1, HIDDEN_),
                      padding_mode=ct.PaddingMode.ZERO)
        rhs_1d = ct.reshape(rhs, (HIDDEN_,))

        scale = ct.load(scale_ptr, index=(row,), shape=(1,),
                        padding_mode=ct.PaddingMode.ZERO)
        scale_s = ct.reshape(scale, ())

        weighted = clone * weight_1d
        row_sum = ct.sum(weighted)   # scalar
        row_dot = ct.sum(weighted * rhs_1d)  # scalar
        centered = weighted * HIDDEN_ - row_sum - rhs_1d * row_dot
        grad = scale_s * centered  # (HIDDEN_,)

        # Store grad_out (mul_6).
        grad_2d = ct.reshape(grad, (1, HIDDEN_))
        ct.store(grad_out_ptr, index=(row, 0), tile=grad_2d)

        # Compute bf16 output.
        # Triton does: dropped = (grad_f32 * keep_scaled_f32).to(bf16)  -- USES F32 grad!
        # Separately: for the sum_5 partial, it uses to_bf16(grad) * to_bf16(keep_scaled)
        keep1 = ct.astype(
            ct.load(keep1_ptr, index=(row, 0), shape=(1, HIDDEN_),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        keep1_1d = ct.reshape(keep1, (HIDDEN_,))
        keep_scaled = keep1_1d * DROP_SCALE_
        keep_scaled_bf = ct.astype(keep_scaled, ct.bfloat16)
        grad_bf = ct.astype(grad, ct.bfloat16)
        # bf16_out uses f32 arithmetic:
        dropped = ct.astype(grad * keep_scaled, ct.bfloat16)
        dropped_2d = ct.reshape(dropped, (1, HIDDEN_))
        ct.store(bf16_out_ptr, index=(row, 0), tile=dropped_2d)
        # bf16 side sum uses bf16-rounded operands:
        dropped_for_sum = ct.astype(
            ct.astype(grad_bf, ct.float32) * ct.astype(keep_scaled_bf, ct.float32),
            ct.bfloat16,
        )

        # Column partials
        acc_clone_rhs = acc_clone_rhs + clone * rhs_1d
        acc_clone = acc_clone + clone
        acc_bf16 = acc_bf16 + ct.astype(dropped_for_sum, ct.float32)

    ct.store(partial_x_normed_ptr, index=(group, 0),
             tile=ct.reshape(acc_clone_rhs, (1, HIDDEN_)))
    ct.store(partial_x_ptr, index=(group, 0),
             tile=ct.reshape(acc_clone, (1, HIDDEN_)))
    ct.store(partial_bf16_ptr, index=(group, 0),
             tile=ct.reshape(acc_bf16, (1, HIDDEN_)))


@ct.kernel
def _finalize_partials_kernel(
    partial_x_normed_ptr,
    partial_x_ptr,
    partial_bf16_ptr,
    out_x_normed_ptr,  # f32 [HIDDEN]
    out_x_ptr,         # f32 [HIDDEN]
    out_bf16_sum_ptr,  # f32 [HIDDEN]
    NUM_GROUPS: ct.Constant[int],
    NUM_GROUPS_PAD: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
):
    p_x_normed = ct.load(partial_x_normed_ptr, index=(0, 0),
                         shape=(NUM_GROUPS_PAD, HIDDEN_),
                         padding_mode=ct.PaddingMode.ZERO)
    p_x = ct.load(partial_x_ptr, index=(0, 0),
                  shape=(NUM_GROUPS_PAD, HIDDEN_),
                  padding_mode=ct.PaddingMode.ZERO)
    p_bf = ct.load(partial_bf16_ptr, index=(0, 0),
                   shape=(NUM_GROUPS_PAD, HIDDEN_),
                   padding_mode=ct.PaddingMode.ZERO)
    idx = ct.arange(NUM_GROUPS_PAD, dtype=ct.int32)
    valid = ct.reshape(idx < NUM_GROUPS, (NUM_GROUPS_PAD, 1))
    zeros = ct.full((NUM_GROUPS_PAD, HIDDEN_), 0.0, dtype=ct.float32)
    p_x_normed_m = ct.where(valid, p_x_normed, zeros)
    p_x_m = ct.where(valid, p_x, zeros)
    p_bf_m = ct.where(valid, p_bf, zeros)

    s_x_normed = ct.sum(p_x_normed_m, axis=0)
    s_x = ct.sum(p_x_m, axis=0)
    s_bf = ct.sum(p_bf_m, axis=0)

    # bf16 round s_bf
    s_bf_rounded = ct.astype(ct.astype(s_bf, ct.bfloat16), ct.float32)

    ct.store(out_x_normed_ptr, index=(0,), tile=s_x_normed)
    ct.store(out_x_ptr, index=(0,), tile=s_x)
    ct.store(out_bf16_sum_ptr, index=(0,), tile=s_bf_rounded)


@oracle_impl(hardware="B200", point="f3ef90ca", ROWS_PER_GROUP=16)
def oracle_forward(inputs, *, ROWS_PER_GROUP: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, shape0, shape1, shape2 = inputs
    device = arg0_1.device

    # Layout (matches Triton):
    # source (arg0) is bf16; view [SEGMENTS=16, SEQ=512, HIDDEN=1024]. The kernel
    # accesses per-row via source_rows = segment * SEQ + seq, so we pass it as
    # a [SEGMENTS, SEQ, HIDDEN] tensor and index (segment, seq, 0).
    view_shape = _shape(shape0)  # [16, 512, 1024]
    source_bshi = arg0_1.view(view_shape).contiguous()  # bf16 [SEGMENTS, SEQ, HIDDEN]

    # keep0, rhs, keep1 are all [ROWS, HIDDEN] via the offsets scheme
    # (rows[:, None] * HIDDEN + cols[None, :]).
    keep0 = arg1_1.contiguous().view(ROWS, HIDDEN)   # dropout mask (f32)
    weight = arg2_1.contiguous().view(HIDDEN)        # f32 [HIDDEN]
    rhs_flat = arg3_1.contiguous().view(ROWS, HIDDEN)  # f32 [ROWS, HIDDEN]
    scale = arg4_1.contiguous().view(ROWS)           # f32 [ROWS]
    keep1 = arg5_1.contiguous().view(ROWS, HIDDEN)   # dropout mask

    num_groups = (ROWS + ROWS_PER_GROUP - 1) // ROWS_PER_GROUP
    num_groups_pad = _next_pow2(num_groups)

    # grad_out is [ROWS, HIDDEN] (flat) — will reshape at end.
    grad_out_flat = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.float32)
    bf16_out = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.bfloat16)
    partial_x_normed = torch.empty((num_groups, HIDDEN), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_groups, HIDDEN), device=device, dtype=torch.float32)
    partial_bf16 = torch.empty((num_groups, HIDDEN), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_groups, 1, 1),
        _row_partials_kernel,
        (source_bshi, keep0, weight, rhs_flat, scale, keep1,
         grad_out_flat, bf16_out,
         partial_x_normed, partial_x, partial_bf16,
         ROWS_PER_GROUP, HIDDEN, ROWS, SEGMENTS, DROP_SCALE),
    )
    # Reshape grad_out to the expected [SEQ, SEGMENTS, HIDDEN].
    grad_out = grad_out_flat.view(SEQ, SEGMENTS, HIDDEN)

    sum_3 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    sum_4 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    sum_5_flat = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    ct.launch(
        stream, (1, 1, 1),
        _finalize_partials_kernel,
        (partial_x_normed, partial_x, partial_bf16,
         sum_3, sum_4, sum_5_flat,
         num_groups, num_groups_pad, HIDDEN),
    )

    # Return values:
    # mul_6 is grad_out
    # sum_3, sum_4 as above
    # view_1 = mul_9_3d.view(shape1) — shape1 is the shape of bf16_out for return
    # permute_1 = view_1.permute(1, 0)
    # convert_element_type_5 = to_f32(to_bf16(sum_5))
    view_1 = bf16_out.view(_shape(shape1))
    permute_1 = view_1.permute(1, 0)
    view_2 = sum_5_flat.view(_shape(shape2))
    # sum_5_flat is already bf16-rounded then f32 in the kernel.
    convert_element_type_5 = view_2

    return grad_out, sum_3, sum_4, view_1, permute_1, convert_element_type_5
