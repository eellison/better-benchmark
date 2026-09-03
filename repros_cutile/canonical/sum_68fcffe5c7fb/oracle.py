"""cuTile port of sum_68fcffe5c7fb: XGLM softmax-backward row kernel.

Matches Triton's flat 1D grid over n_rows with BLOCK_M rows per block.
Bias is broadcast over heads: bias offset uses (outer, 0, query, cols).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


DROPOUT_SCALE_F32 = 1.1111111640930176
F32_MIN = -3.4028234663852886e38


@ct.kernel
def _xglm_softmax_backward_kernel(
    grad_ptr,          # bf16 flat [outer * heads * q_len * k_len]
    keep_ptr,          # u8   flat
    score_ptr,         # bf16 flat
    bias_ptr,          # f32 flat [outer * q_len * k_len]  (heads dim=1)
    row_shift_ptr,     # f32 [n_rows]
    row_denom_ptr,     # f32 [n_rows]
    out_ptr,           # bf16 flat
    n_rows: ct.Constant[int],
    heads: ct.Constant[int],
    q_len: ct.Constant[int],
    k_len: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    DROPOUT_SCALE: ct.Constant[float],
    FLOOR: ct.Constant[float],
):
    block = ct.bid(0)
    rows_1d = block * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)
    cols_1d = ct.arange(BLOCK_N, dtype=ct.int32)
    row_valid = rows_1d < n_rows
    col_valid = cols_1d < k_len

    # For each row: outer = row // (heads * q_len), query = row % q_len
    flat_bh = rows_1d // q_len
    outer = flat_bh // heads
    query = rows_1d - flat_bh * q_len
    # bias offsets = outer * (q_len * k_len) + query * k_len + col
    rows_2d = ct.reshape(rows_1d, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols_1d, (1, BLOCK_N))
    outer_2d = ct.reshape(outer, (BLOCK_M, 1))
    query_2d = ct.reshape(query, (BLOCK_M, 1))
    rows_bc = ct.broadcast_to(rows_2d, (BLOCK_M, BLOCK_N))
    cols_bc = ct.broadcast_to(cols_2d, (BLOCK_M, BLOCK_N))
    outer_bc = ct.broadcast_to(outer_2d, (BLOCK_M, BLOCK_N))
    query_bc = ct.broadcast_to(query_2d, (BLOCK_M, BLOCK_N))
    bias_offsets = outer_bc * (q_len * k_len) + query_bc * k_len + cols_bc

    offsets = rows_bc * k_len + cols_bc

    grad = ct.astype(ct.gather(grad_ptr, (offsets,)), ct.float32)
    keep = ct.astype(ct.gather(keep_ptr, (offsets,)), ct.float32)
    score_base = ct.astype(ct.gather(score_ptr, (offsets,)), ct.float32)
    bias = ct.gather(bias_ptr, (bias_offsets,))

    add = score_base + bias
    floor_tile = ct.full((BLOCK_M, BLOCK_N), FLOOR, dtype=ct.float32)
    floored = ct.maximum(add, floor_tile)

    row_shift = ct.astype(ct.load(row_shift_ptr, index=(block,), shape=(BLOCK_M,)),
                          ct.float32)
    row_denom = ct.astype(ct.load(row_denom_ptr, index=(block,), shape=(BLOCK_M,)),
                          ct.float32)
    row_shift_2d = ct.reshape(row_shift, (BLOCK_M, 1))
    row_denom_2d = ct.reshape(row_denom, (BLOCK_M, 1))

    probs = ct.exp(floored - row_shift_2d) / row_denom_2d

    scaled_keep = keep * DROPOUT_SCALE
    scaled_grad = grad * scaled_keep
    product = scaled_grad * probs
    row_sum = ct.sum(product, axis=1, keepdims=True)
    fma = -probs * row_sum + product
    half = fma * 0.5

    is_eq = add == floor_tile
    selected_eq = ct.where(is_eq, half, fma)
    is_lt = add < floor_tile
    zero_tile = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
    selected = ct.where(is_lt, zero_tile, selected_eq)
    out_bf = ct.astype(selected, ct.bfloat16)

    ct.scatter(out_ptr, (offsets,), out_bf)


@oracle_impl(hardware="B200", point="8da16745", BLOCK_M=2, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N):
    (
        arg0_1,  # bf16 [512, 128, 128]
        arg1_1,  # b8 [512, 128, 128]
        arg2_1,  # bf16 [512, 128, 128]
        arg3_1,  # f32 [32, 1, 128, 128]  bias
        arg4_1,  # f32 [512, 128, 1]
        arg5_1,  # f32 [512, 128, 1]
        shape0,
        _shape1,
        _shape2,
        shape3,
    ) = inputs
    full_shape = tuple(int(dim) for dim in shape0)
    out_shape = tuple(int(dim) for dim in shape3)
    outer, heads, q_len, k_len = full_shape
    n_rows = outer * heads * q_len

    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    # Flat views
    grad_flat = arg0_1.view(-1)
    keep_flat = arg1_1.view(torch.uint8).view(-1)
    score_flat = arg2_1.view(-1)
    bias_flat = arg3_1.reshape(-1)  # length outer * q_len * k_len (heads=1 collapses)
    row_shift_1d = arg4_1.view(n_rows)
    row_denom_1d = arg5_1.view(n_rows)
    out_flat = out.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _xglm_softmax_backward_kernel,
        (
            grad_flat, keep_flat, score_flat, bias_flat,
            row_shift_1d, row_denom_1d, out_flat,
            n_rows, heads, q_len, k_len, BLOCK_M, BLOCK_N,
            DROPOUT_SCALE_F32, F32_MIN,
        ),
    )
    return out
