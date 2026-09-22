"""cuTile port of sum_sum_sum_65c3d9f36fd9: MegatronBERT LayerNorm-backward tail.

Similar to 127fc8edd5da but with three bf16 inputs added, and no bf16-flat side
output (returns bf16 permute view of the dropout-mul_7 output).

Strategy: torch does the exact elementwise ops; cuTile kernels handle column
reductions [8192, 1024] → [1024].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BLOCK_C = 1024


@ct.kernel
def _col_sum_kernel(
    x_ptr, out_ptr,
    ROWS_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    ROWS_PER_BLOCK: ct.Constant[int],
):
    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    col_mask = cols < HIDDEN_
    acc = ct.zeros((BLOCK_C,), dtype=ct.float32)

    num_row_tiles = ct.cdiv(ROWS_, ROWS_PER_BLOCK)
    for row_tile in range(num_row_tiles):
        row_start = row_tile * ROWS_PER_BLOCK
        for local_row in ct.static_iter(range(ROWS_PER_BLOCK)):
            row = row_start + local_row
            if row < ROWS_:
                x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_C),
                             padding_mode=ct.PaddingMode.ZERO)
                x_1d = ct.reshape(x, (BLOCK_C,))
                acc = acc + ct.where(col_mask, x_1d, 0.0)

    ct.store(out_ptr, index=(0,), tile=acc)


@ct.kernel
def _col_sum_bf16_kernel(
    x_ptr, out_ptr,
    ROWS_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    ROWS_PER_BLOCK: ct.Constant[int],
):
    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    col_mask = cols < HIDDEN_
    acc = ct.zeros((BLOCK_C,), dtype=ct.float32)

    num_row_tiles = ct.cdiv(ROWS_, ROWS_PER_BLOCK)
    for row_tile in range(num_row_tiles):
        row_start = row_tile * ROWS_PER_BLOCK
        for local_row in ct.static_iter(range(ROWS_PER_BLOCK)):
            row = row_start + local_row
            if row < ROWS_:
                x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_C),
                             padding_mode=ct.PaddingMode.ZERO)
                x_f = ct.astype(x, ct.float32)
                x_1d = ct.reshape(x_f, (BLOCK_C,))
                acc = acc + ct.where(col_mask, x_1d, 0.0)

    ct.store(out_ptr, index=(0,), tile=acc)


@oracle_impl(hardware="B200", point="e3ecf7e0")
def oracle_forward(inputs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
        shape0, shape1, shape2, shape3, shape4,
    ) = inputs
    full_shape = tuple(int(d) for d in shape0)  # [16, 512, 1024]
    flat_shape = tuple(int(d) for d in shape3)  # [8192, 1024]
    sum_shape = tuple(int(d) for d in shape4)   # [1024]
    device = arg0_1.device
    rows = int(flat_shape[0])
    hidden = int(flat_shape[1])

    view0 = arg0_1.to(torch.float32).view(full_shape)
    view1 = arg1_1.to(torch.float32).view(full_shape)
    view2 = arg2_1.to(torch.float32).view(full_shape)
    add = view0 + view1
    add_1 = add + view2  # f32
    mul = add_1 * arg3_1
    mul_1 = mul * 1024
    sum_1 = mul.sum(dim=2, keepdim=True)
    mul_2 = mul * arg4_1
    sum_2 = mul_2.sum(dim=2, keepdim=True)
    mul_3 = arg4_1 * sum_2
    sub_1 = (mul_1 - sum_1) - mul_3
    mul_4 = arg5_1 * sub_1
    add_2 = arg6_1 + mul_4  # f32
    ct_3 = add_2.to(torch.bfloat16)
    ct_4 = arg7_1.to(torch.bfloat16)
    mul_6 = ct_4 * 1.1111111111111112
    mul_7 = ct_3 * mul_6  # bf16

    # Match strided outputs
    add_2_strided = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=device, dtype=torch.float32,
    )
    add_2_strided.copy_(add_2)

    view_3 = torch.empty_strided(
        flat_shape,
        (flat_shape[1], 1),
        device=device, dtype=torch.bfloat16,
    )
    view_3.copy_(mul_7.view(flat_shape))

    permute = view_3.permute(1, 0)

    mul_5 = add_1 * arg4_1
    mul_5_2d = mul_5.reshape(rows, hidden).contiguous()
    add_1_2d = add_1.reshape(rows, hidden).contiguous()

    out_sum3 = torch.empty((BLOCK_C,), device=device, dtype=torch.float32)
    out_sum4 = torch.empty((BLOCK_C,), device=device, dtype=torch.float32)
    out_bf16 = torch.empty((BLOCK_C,), device=device, dtype=torch.float32)

    ROWS_PER_BLOCK = 16
    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _col_sum_kernel,
              (mul_5_2d, out_sum3, rows, hidden, ROWS_PER_BLOCK))
    ct.launch(stream, (1, 1, 1), _col_sum_kernel,
              (add_1_2d, out_sum4, rows, hidden, ROWS_PER_BLOCK))
    ct.launch(stream, (1, 1, 1), _col_sum_bf16_kernel,
              (view_3, out_bf16, rows, hidden, ROWS_PER_BLOCK))

    sum_3 = torch.empty_strided(sum_shape, (1,), device=device, dtype=torch.float32)
    sum_3.copy_(out_sum3[:hidden])
    sum_4 = torch.empty_strided(sum_shape, (1,), device=device, dtype=torch.float32)
    sum_4.copy_(out_sum4[:hidden])
    out6 = torch.empty_strided(sum_shape, (1,), device=device, dtype=torch.float32)
    out6.copy_(out_bf16[:hidden].to(torch.bfloat16).to(torch.float32))

    return sum_3, sum_4, add_2_strided, view_3, permute, out6
