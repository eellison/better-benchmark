"""cuTile port of sum_sum_sum_2f5510748eeb: GPTNeo LN-backward + embedding-backward.

cuTile kernels:
- `_ln_backward_kernel`: per-row LayerNorm-backward math producing mul_6 and add_3.
- `_col_partial_reduce_kernel`: split-K partial column reductions over ROWS for
  mul_6 (feeds sum_3) and add_1 (feeds sum_4).
- `_col_finalize_reduce_kernel`: reduces per-chunk partials into sum_3/sum_4.
- `_batch_reduce_kernel`: reduces add_3 over the BATCH axis to produce sum_5.

Torch epilogue does the masked_index_put_accumulate scatters (position and
vocab tables) because cuTile lacks a masked atomic-add scatter.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 128
ROWS = BATCH * SEQ
HIDDEN = 2048


@ct.kernel
def _ln_backward_kernel(
    add_1_ptr,     # f32 [ROWS, HIDDEN]
    weight_ptr,    # f32 [HIDDEN]
    sub_input_ptr, # f32 [ROWS, HIDDEN]  (arg5 + arg6 - arg7)
    invstd_ptr,    # f32 [ROWS]
    resid_ptr,     # f32 [ROWS, HIDDEN]  (arg9_1)
    mul_6_ptr,     # f32 [ROWS, HIDDEN]  output — add_1 * mul_2
    add_3_ptr,     # f32 [ROWS, HIDDEN]  output — resid + mul_5
    HIDDEN_: ct.Constant[int],
):
    row = ct.bid(0)
    add_1 = ct.load(add_1_ptr, index=(row, 0), shape=(1, HIDDEN_))
    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_,))
    weight_2d = ct.reshape(weight, (1, HIDDEN_))
    sub_input = ct.load(sub_input_ptr, index=(row, 0), shape=(1, HIDDEN_))
    invstd = ct.load(invstd_ptr, index=(row,), shape=(1,))
    resid = ct.load(resid_ptr, index=(row, 0), shape=(1, HIDDEN_))

    mul_val = add_1 * weight_2d
    mul_1 = mul_val * HIDDEN_
    sum_1 = ct.sum(mul_val)
    mul_2 = sub_input * invstd
    mul_3 = mul_val * mul_2
    sum_2 = ct.sum(mul_3)
    mul_4 = mul_2 * sum_2
    sub_1 = mul_1 - sum_1
    sub_2 = sub_1 - mul_4
    div = invstd * (1.0 / HIDDEN_)
    mul_5 = div * sub_2
    add_3 = resid + mul_5

    # mul_6 is add_1 * mul_2 (Repro semantics).
    mul_6 = add_1 * mul_2
    ct.store(mul_6_ptr, index=(row, 0), tile=mul_6)
    ct.store(add_3_ptr, index=(row, 0), tile=add_3)


@ct.kernel
def _col_partial_reduce_kernel(
    mul_6_ptr,        # f32 [ROWS, HIDDEN]
    add_1_ptr,        # f32 [ROWS, HIDDEN]
    partial_sum3_ptr, # f32 [num_r_tiles, HIDDEN]
    partial_sum4_ptr, # f32 [num_r_tiles, HIDDEN]
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    r_block = ct.bid(1)

    mul_6 = ct.load(mul_6_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C))
    add_1 = ct.load(add_1_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C))

    s3 = ct.sum(mul_6, axis=0)  # (BLOCK_C,)
    s4 = ct.sum(add_1, axis=0)

    ct.store(partial_sum3_ptr, index=(r_block, c_block),
             tile=ct.reshape(s3, (1, BLOCK_C)))
    ct.store(partial_sum4_ptr, index=(r_block, c_block),
             tile=ct.reshape(s4, (1, BLOCK_C)))


@ct.kernel
def _col_finalize_reduce_kernel(
    partial_sum3_ptr, # f32 [NUM_R_TILES, HIDDEN]
    partial_sum4_ptr, # f32 [NUM_R_TILES, HIDDEN]
    sum3_ptr,         # f32 [HIDDEN]
    sum4_ptr,         # f32 [HIDDEN]
    NUM_R_TILES: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    p3 = ct.load(partial_sum3_ptr, index=(0, c_block),
                 shape=(NUM_R_TILES, BLOCK_C))
    p4 = ct.load(partial_sum4_ptr, index=(0, c_block),
                 shape=(NUM_R_TILES, BLOCK_C))
    s3 = ct.sum(p3, axis=0)
    s4 = ct.sum(p4, axis=0)
    ct.store(sum3_ptr, index=(c_block,), tile=s3)
    ct.store(sum4_ptr, index=(c_block,), tile=s4)


@ct.kernel
def _batch_reduce_kernel(
    add_3_ptr,     # f32 [BATCH, SEQ, HIDDEN]
    sum_5_ptr,     # f32 [SEQ, HIDDEN]
    BLOCK_C: ct.Constant[int],
):
    s_block = ct.bid(0)  # seq index (1 per block)
    c_block = ct.bid(1)  # hidden block
    tile = ct.load(add_3_ptr, index=(0, s_block, c_block),
                   shape=(BATCH, 1, BLOCK_C))
    reduced = ct.sum(tile, axis=0)  # (1, BLOCK_C)
    ct.store(sum_5_ptr, index=(s_block, c_block), tile=reduced)


@oracle_impl(hardware="B200", point="25bd45b5")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
     arg7_1, arg8_1, arg9_1, arg10_1, arg11_1,
     shape0, shape1, shape2, shape3, shape4, shape5) = inputs
    device = arg0_1.device

    view = arg1_1.view(BATCH, SEQ, HIDDEN)
    view_1 = arg2_1.view(BATCH, SEQ, HIDDEN)
    view_2 = arg3_1.view(BATCH, SEQ, HIDDEN)
    add_1 = view.to(torch.float32) + view_1.to(torch.float32) + view_2.to(torch.float32)
    sub_input = (arg5_1 + arg6_1) - arg7_1
    invstd_flat = arg8_1.view(-1).contiguous()

    add_1_flat = add_1.view(ROWS, HIDDEN).contiguous()
    sub_input_flat = sub_input.view(ROWS, HIDDEN).contiguous()
    resid_flat = arg9_1.view(ROWS, HIDDEN).contiguous()

    mul_6_out = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.float32)
    add_3_out = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ROWS, 1, 1), _ln_backward_kernel,
        (add_1_flat, arg4_1, sub_input_flat, invstd_flat, resid_flat,
         mul_6_out, add_3_out, HIDDEN),
    )

    # Column reductions for sum_3 and sum_4 (split-K over ROWS).
    BLOCK_R_PART = 256
    BLOCK_C_PART = 128
    NUM_R_TILES = ROWS // BLOCK_R_PART  # 16

    partial_sum3 = torch.empty((NUM_R_TILES, HIDDEN), device=device, dtype=torch.float32)
    partial_sum4 = torch.empty((NUM_R_TILES, HIDDEN), device=device, dtype=torch.float32)

    ct.launch(
        stream, (HIDDEN // BLOCK_C_PART, NUM_R_TILES, 1),
        _col_partial_reduce_kernel,
        (mul_6_out, add_1_flat, partial_sum3, partial_sum4,
         BLOCK_R_PART, BLOCK_C_PART),
    )

    sum_3 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    sum_4 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    ct.launch(
        stream, (HIDDEN // BLOCK_C_PART, 1, 1),
        _col_finalize_reduce_kernel,
        (partial_sum3, partial_sum4, sum_3, sum_4,
         NUM_R_TILES, BLOCK_C_PART),
    )

    # Batch reduction for sum_5: reduce add_3 over BATCH.
    add_3_bch = add_3_out.view(BATCH, SEQ, HIDDEN)
    BLOCK_C_BATCH = 128
    sum_5_2d = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    ct.launch(
        stream, (SEQ, HIDDEN // BLOCK_C_BATCH, 1),
        _batch_reduce_kernel,
        (add_3_bch, sum_5_2d, BLOCK_C_BATCH),
    )
    sum_5 = sum_5_2d.view(1, SEQ, HIDDEN)

    # Position table scatter: full_1[arg10_1] += sum_5 (all-True mask)
    shape_p3 = tuple(int(x) for x in shape3)  # [1, 128, 1]
    shape_p4 = tuple(int(x) for x in shape4)  # [2048, 2048]
    shape_p5 = tuple(int(x) for x in shape5)  # [50257, 2048]

    full_pos = torch.zeros(shape_p4, device=device, dtype=torch.float32)
    mask_all = torch.ones(shape_p3, device=device, dtype=torch.bool)
    pos_result = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
        full_pos, mask_all, [arg10_1], sum_5,
    )

    # Token table scatter: full_2[arg11_1] += add_3 with mask
    ge = arg11_1 >= 0
    lt = arg11_1 < 50257
    ne = arg11_1 != -1
    mask_token = (ge & lt & ne).unsqueeze(-1)
    full_tok = torch.zeros(shape_p5, device=device, dtype=torch.float32)
    tok_result = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
        full_tok, mask_token, [arg11_1], add_3_bch,
    )

    # add_4 = slice(arg0_1, 0, 0, -7).to(f32) + tok_result
    slice_1 = arg0_1[0:-7]
    add_4 = slice_1.to(torch.float32) + tok_result

    return sum_3, sum_4, pos_result, add_4
