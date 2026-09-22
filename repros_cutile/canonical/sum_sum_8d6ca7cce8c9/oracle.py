"""cuTile port of sum_sum_8d6ca7cce8c9: ConvBERT reshape/permute/sum tail.

Row-blocked kernel computes the two bf16 pointwise outputs (view_2, mul_1)
and partial column sums. Second kernel finalizes both column reductions
into [384] with the required bf16 rounding boundary for the first sum.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 16384
HIDDEN = 384
SEQ = 512


@ct.kernel
def _row_partials_kernel(
    view_ptr,       # bf16 [ROWS, HIDDEN_PAD]  (padded dense)
    rhs_bf_ptr,     # bf16 [ROWS, HIDDEN_PAD]
    clone_ptr,      # bf16 [ROWS, HIDDEN_PAD]
    arg4_ptr,       # bf16 [ROWS, HIDDEN_PAD]
    add_ptr,        # bf16 [ROWS, HIDDEN_PAD]
    mul_ptr,        # bf16 [ROWS, HIDDEN_PAD]
    partial_add_ptr, # f32 [num_groups, HIDDEN_PAD]
    partial_mul_ptr, # f32 [num_groups, HIDDEN_PAD]
    ROW_BLOCK: ct.Constant[int],
    HIDDEN_PAD: ct.Constant[int],
):
    row_group = ct.bid(0)
    v = ct.load(view_ptr, index=(row_group, 0), shape=(ROW_BLOCK, HIDDEN_PAD))
    r = ct.load(rhs_bf_ptr, index=(row_group, 0), shape=(ROW_BLOCK, HIDDEN_PAD))
    c = ct.load(clone_ptr, index=(row_group, 0), shape=(ROW_BLOCK, HIDDEN_PAD))
    a4 = ct.load(arg4_ptr, index=(row_group, 0), shape=(ROW_BLOCK, HIDDEN_PAD))

    v_f = ct.astype(v, ct.float32)
    r_f = ct.astype(r, ct.float32)
    c_f = ct.astype(c, ct.float32)
    a4_f = ct.astype(a4, ct.float32)

    mul_bf = ct.astype(v_f * r_f, ct.bfloat16)
    add_bf = ct.astype(ct.astype(mul_bf, ct.float32) + c_f, ct.bfloat16)
    mul1_bf = ct.astype(v_f * a4_f, ct.bfloat16)

    ct.store(add_ptr, index=(row_group, 0), tile=add_bf)
    ct.store(mul_ptr, index=(row_group, 0), tile=mul1_bf)

    partial_add = ct.sum(ct.astype(add_bf, ct.float32), axis=0, keepdims=True)
    partial_mul = ct.sum(ct.astype(mul1_bf, ct.float32), axis=0, keepdims=True)
    ct.store(partial_add_ptr, index=(row_group, 0), tile=partial_add)
    ct.store(partial_mul_ptr, index=(row_group, 0), tile=partial_mul)


@ct.kernel
def _finalize_kernel(
    partial_add_ptr,   # f32 [num_groups, HIDDEN_PAD]
    partial_mul_ptr,   # f32 [num_groups, HIDDEN_PAD]
    sum_add_ptr,       # f32 [HIDDEN_PAD]
    sum_mul_ptr,       # f32 [HIDDEN_PAD]
    NUM_GROUPS: ct.Constant[int],
    HIDDEN_PAD_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    col_block = ct.bid(0)
    p_add = ct.load(partial_add_ptr, index=(0, col_block),
                    shape=(NUM_GROUPS, BLOCK_H))
    p_mul = ct.load(partial_mul_ptr, index=(0, col_block),
                    shape=(NUM_GROUPS, BLOCK_H))
    total_add = ct.sum(p_add, axis=0)   # (BLOCK_H,)
    total_mul = ct.sum(p_mul, axis=0)   # (BLOCK_H,)
    # bf16 rounding for sum_add (matches Repro's convert_element_type_1/2 chain)
    total_add_bf16_bf = ct.astype(total_add, ct.bfloat16)
    total_add_round = ct.astype(total_add_bf16_bf, ct.float32)
    ct.store(sum_add_ptr, index=(col_block,), tile=total_add_round)
    ct.store(sum_mul_ptr, index=(col_block,), tile=total_mul)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


HIDDEN_PAD = 512  # next power of 2 >= 384


@oracle_impl(hardware="B200", point="0c475f9a", ROW_BLOCK=64)
def oracle_forward(inputs, *, ROW_BLOCK: int):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1,
     _shape_param_0, _shape_param_1, shape_view2, shape_sum0, shape_sum1) = inputs
    device = arg1_1.device

    # Prepare padded dense inputs (torch pre-processing).
    view = arg1_1.view(ROWS, HIDDEN)
    rhs = (arg2_1.float() + arg3_1.float()).to(torch.bfloat16).permute(0, 2, 1).contiguous().view(ROWS, HIDDEN)
    clone = arg0_1.permute(0, 2, 1, 3).contiguous().view(ROWS, HIDDEN)
    arg4_2d = arg4_1.view(ROWS, HIDDEN)

    # Pad HIDDEN 384 -> HIDDEN_PAD 512 with zeros.
    def _pad_hidden(x):
        pad = torch.zeros((ROWS, HIDDEN_PAD), device=device, dtype=x.dtype)
        pad[:, :HIDDEN] = x
        return pad

    view_p = _pad_hidden(view)
    rhs_p = _pad_hidden(rhs)
    clone_p = _pad_hidden(clone)
    arg4_p = _pad_hidden(arg4_2d)

    view_2 = torch.empty_strided(
        _shape_tuple(shape_view2), (HIDDEN, 1),
        device=device, dtype=torch.bfloat16,
    )
    out_mul = torch.empty_strided(
        _shape_tuple(_shape_param_0), (SEQ * HIDDEN, HIDDEN, 1),
        device=device, dtype=torch.bfloat16,
    )
    sum_add = torch.empty_strided(
        _shape_tuple(shape_sum0), (1,), device=device, dtype=torch.float32,
    )
    sum_mul = torch.empty_strided(
        _shape_tuple(shape_sum1), (1, 1), device=device, dtype=torch.float32,
    )

    num_groups = ROWS // ROW_BLOCK
    partial_add = torch.empty((num_groups, HIDDEN_PAD), device=device, dtype=torch.float32)
    partial_mul = torch.empty((num_groups, HIDDEN_PAD), device=device, dtype=torch.float32)
    add_pad = torch.empty((ROWS, HIDDEN_PAD), device=device, dtype=torch.bfloat16)
    mul_pad = torch.empty((ROWS, HIDDEN_PAD), device=device, dtype=torch.bfloat16)
    sum_add_pad = torch.empty(HIDDEN_PAD, device=device, dtype=torch.float32)
    sum_mul_pad = torch.empty(HIDDEN_PAD, device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_groups, 1, 1),
        _row_partials_kernel,
        (view_p, rhs_p, clone_p, arg4_p,
         add_pad, mul_pad,
         partial_add, partial_mul,
         ROW_BLOCK, HIDDEN_PAD),
    )
    FINAL_BLOCK_H = 16
    ct.launch(
        stream, (HIDDEN_PAD // FINAL_BLOCK_H, 1, 1),
        _finalize_kernel,
        (partial_add, partial_mul, sum_add_pad, sum_mul_pad,
         num_groups, HIDDEN_PAD, FINAL_BLOCK_H),
    )
    # Slice back to HIDDEN=384
    view_2.view(ROWS, HIDDEN).copy_(add_pad[:, :HIDDEN])
    out_mul.view(ROWS, HIDDEN).copy_(mul_pad[:, :HIDDEN])
    sum_add.view(HIDDEN).copy_(sum_add_pad[:HIDDEN])
    sum_mul.view(HIDDEN).copy_(sum_mul_pad[:HIDDEN])

    permute_2 = view_2.permute(1, 0)
    permute_3 = out_mul.permute(0, 2, 1)
    return view_2, permute_2, sum_add, permute_3, sum_mul
