"""cuTile port of sum_sum_sum_8a3378108574: GPTNeo LayerNorm-backward tail.

Mirrors the Triton multi-kernel plan:
  - _row_store_and_partial_reduce_kernel: per-row-block, compute row-local
    reductions, store dense f32 update + bf16 side, emit per-row-block
    column partials for the three column reductions.
  - _finalize_column_partials_kernel: column reductions from partials.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 2048
ROW_SPLIT = 4
FINAL_BLOCK_H = 16


@ct.kernel
def _row_store_and_partial_reduce_kernel(
    a_ptr,          # bf16 [rows, hidden]
    b_ptr,          # bf16 [rows, hidden]
    c_ptr,          # bf16 [rows, hidden]
    gamma_ptr,      # f32 [hidden]
    rhs_ptr,        # f32 [rows, hidden]
    row_scale_ptr,  # f32 [rows]
    residual_ptr,   # f32 [rows, hidden]
    add_out_ptr,    # f32 [rows, hidden]
    bf16_out_ptr,   # bf16 [rows, hidden]
    partial_x_rhs_ptr,     # f32 [num_row_blocks, hidden]
    partial_x_ptr,         # f32 [num_row_blocks, hidden]
    partial_bf16_out_ptr,  # f32 [num_row_blocks, hidden]
    ROWS_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    ROW_SPLIT_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    pid = ct.bid(0)

    gamma = ct.load(gamma_ptr, index=(0,), shape=(BLOCK_H,))
    gamma_2d = ct.reshape(gamma, (1, BLOCK_H))

    acc_x_rhs = ct.zeros((BLOCK_H,), dtype=ct.float32)
    acc_x = ct.zeros((BLOCK_H,), dtype=ct.float32)
    acc_bf16_out = ct.zeros((BLOCK_H,), dtype=ct.float32)

    for r in range(ROW_SPLIT_):
        row = pid * ROW_SPLIT_ + r

        a_bf = ct.load(a_ptr, index=(row, 0), shape=(1, BLOCK_H))
        b_bf = ct.load(b_ptr, index=(row, 0), shape=(1, BLOCK_H))
        c_bf = ct.load(c_ptr, index=(row, 0), shape=(1, BLOCK_H))
        x = ct.astype(a_bf, ct.float32) + ct.astype(b_bf, ct.float32) + ct.astype(c_bf, ct.float32)

        rhs = ct.load(rhs_ptr, index=(row, 0), shape=(1, BLOCK_H))
        residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
        row_scale = ct.load(row_scale_ptr, index=(row,), shape=(1,))
        row_scale_2d = ct.reshape(row_scale, (1, 1))

        weighted = x * gamma_2d
        row_sum = ct.sum(weighted, axis=1)
        row_dot = ct.sum(weighted * rhs, axis=1)
        row_sum_2d = ct.reshape(row_sum, (1, 1))
        row_dot_2d = ct.reshape(row_dot, (1, 1))
        update = row_scale_2d * (weighted * HIDDEN_ - row_sum_2d - rhs * row_dot_2d)
        out = residual + update
        out_bf = ct.astype(out, ct.bfloat16)

        ct.store(add_out_ptr, index=(row, 0), tile=out)
        ct.store(bf16_out_ptr, index=(row, 0), tile=out_bf)

        # Accumulate per-column partials
        x_rhs_1d = ct.reshape(x * rhs, (BLOCK_H,))
        x_1d = ct.reshape(x, (BLOCK_H,))
        out_bf_f32 = ct.reshape(ct.astype(out_bf, ct.float32), (BLOCK_H,))
        acc_x_rhs = acc_x_rhs + x_rhs_1d
        acc_x = acc_x + x_1d
        acc_bf16_out = acc_bf16_out + out_bf_f32

    ct.store(partial_x_rhs_ptr, index=(pid, 0), tile=ct.reshape(acc_x_rhs, (1, BLOCK_H)))
    ct.store(partial_x_ptr, index=(pid, 0), tile=ct.reshape(acc_x, (1, BLOCK_H)))
    ct.store(partial_bf16_out_ptr, index=(pid, 0), tile=ct.reshape(acc_bf16_out, (1, BLOCK_H)))


@ct.kernel
def _finalize_column_partials_kernel(
    partial_x_rhs_ptr,     # f32 [num_row_blocks, hidden]
    partial_x_ptr,         # f32 [num_row_blocks, hidden]
    partial_bf16_out_ptr,  # f32 [num_row_blocks, hidden]
    out_x_rhs_ptr,   # f32 [hidden]
    out_x_ptr,       # f32 [hidden]
    out_bf16_sum_ptr,  # f32 [hidden]
    NUM_ROW_BLOCKS: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    col_block = ct.bid(0)

    acc_x_rhs = ct.zeros((BLOCK_H,), dtype=ct.float32)
    acc_x = ct.zeros((BLOCK_H,), dtype=ct.float32)
    acc_bf16 = ct.zeros((BLOCK_H,), dtype=ct.float32)

    for g in range(NUM_ROW_BLOCKS):
        xr = ct.load(partial_x_rhs_ptr, index=(g, col_block), shape=(1, BLOCK_H))
        xv = ct.load(partial_x_ptr, index=(g, col_block), shape=(1, BLOCK_H))
        bv = ct.load(partial_bf16_out_ptr, index=(g, col_block), shape=(1, BLOCK_H))
        acc_x_rhs = acc_x_rhs + ct.reshape(xr, (BLOCK_H,))
        acc_x = acc_x + ct.reshape(xv, (BLOCK_H,))
        acc_bf16 = acc_bf16 + ct.reshape(bv, (BLOCK_H,))

    # bf16 roundtrip on final sum (matches Triton)
    bf = ct.astype(acc_bf16, ct.bfloat16)
    ct.store(out_x_rhs_ptr, index=(col_block,), tile=acc_x_rhs)
    ct.store(out_x_ptr, index=(col_block,), tile=acc_x)
    ct.store(out_bf16_sum_ptr, index=(col_block,), tile=ct.astype(bf, ct.float32))


@oracle_impl(hardware="B200", point="b9b3dbcc", BLOCK_H=HIDDEN, FINAL_BLOCK=FINAL_BLOCK_H)
def oracle_forward(inputs, *, BLOCK_H, FINAL_BLOCK):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
        _shape0, _shape1, _shape2, _shape3, _shape4,
    ) = inputs

    view_shape = tuple(int(dim) for dim in _shape0)
    flat_shape = tuple(int(dim) for dim in _shape3)
    vec_shape = tuple(int(dim) for dim in _shape4)
    rows = int(flat_shape[0])
    hidden = int(flat_shape[1])
    num_row_blocks = (rows + ROW_SPLIT - 1) // ROW_SPLIT
    device = arg0_1.device

    a = arg0_1.reshape(rows, hidden)
    b = arg1_1.reshape(rows, hidden)
    c = arg2_1.reshape(rows, hidden)
    rhs = arg4_1.reshape(rows, hidden)
    row_scale = arg5_1.reshape(rows)
    residual = arg6_1.reshape(rows, hidden)

    add_out = torch.empty_strided(
        view_shape,
        tuple(int(s) for s in arg6_1.stride()),
        device=device, dtype=torch.float32,
    )
    add_out_2d = add_out.view(rows, hidden)
    bf16_out = torch.empty_strided(
        flat_shape, (hidden, 1), device=device, dtype=torch.bfloat16,
    )

    partial_x_rhs = torch.empty((num_row_blocks, hidden), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_row_blocks, hidden), device=device, dtype=torch.float32)
    partial_bf16 = torch.empty((num_row_blocks, hidden), device=device, dtype=torch.float32)

    sum_x_rhs = torch.empty(vec_shape, device=device, dtype=torch.float32)
    sum_x = torch.empty(vec_shape, device=device, dtype=torch.float32)
    bf16_sum = torch.empty(vec_shape, device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_row_blocks, 1, 1), _row_store_and_partial_reduce_kernel,
        (a, b, c, arg3_1, rhs, row_scale, residual, add_out_2d, bf16_out,
         partial_x_rhs, partial_x, partial_bf16,
         rows, hidden, ROW_SPLIT, BLOCK_H),
    )
    ct.launch(
        stream, ((hidden + FINAL_BLOCK - 1) // FINAL_BLOCK, 1, 1),
        _finalize_column_partials_kernel,
        (partial_x_rhs, partial_x, partial_bf16,
         sum_x_rhs, sum_x, bf16_sum,
         num_row_blocks, hidden, FINAL_BLOCK),
    )

    return sum_x_rhs, sum_x, add_out, bf16_out, bf16_out.permute(1, 0), bf16_sum
