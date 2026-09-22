"""cuTile port of sum_sum_sum_e7781939b0a2: GPT-2 LayerNorm-backward + dropout tail.

Reference (from repro.py):
  view    = arg0.bf16 → f32 view                f32[32, 512, 768]
  mul     = view * arg1                          (arg1 = weight [768])
  mul_1   = mul * 768
  sum_1   = sum(mul, dim=-1)
  mul_2   = mul * arg2                           (arg2 = xhat)
  sum_2   = sum(mul_2, dim=-1)
  mul_3   = arg2 * sum_2
  sub_1   = (mul_1 - sum_1) - mul_3
  mul_4   = arg3 * sub_1                          (arg3 = scale [rows, 1])
  add     = arg4 + mul_4                          f32 residual add
  ct_1    = add.bf16
  ct_2    = arg5.b8.bf16                          (dropout keep mask)
  mul_6   = ct_2 * 1.1111...                      bf16
  mul_7   = ct_1 * mul_6                          bf16 dropout output
  view_1  = mul_7.view(16384, 768)                bf16
  sum_5   = sum(view_1, dim=0, dtype=f32)         f32
  out4    = sum_5.bf16.f32

Returns: (sum_3, sum_4, add, view_1, out4).

Strategy: torch does the exact elementwise ops; cuTile kernel handles column
reductions [16384, 768] → [768].
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


def _run(inputs, point):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
        shape0, shape1, shape2,
    ) = inputs
    full_shape = tuple(int(d) for d in shape0)  # [B, S, H]
    flat_shape = tuple(int(d) for d in shape1)  # [rows, H]
    sum_shape = tuple(int(d) for d in shape2)   # [H]
    device = arg0_1.device
    rows = int(flat_shape[0])
    hidden = int(flat_shape[1])

    # Elementwise via torch (exact match with eager)
    view = arg0_1.to(torch.float32).view(full_shape)
    mul = view * arg1_1
    mul_1 = mul * 768
    sum_1 = mul.sum(dim=2, keepdim=True)
    mul_2 = mul * arg2_1
    sum_2 = mul_2.sum(dim=2, keepdim=True)
    mul_3 = arg2_1 * sum_2
    sub_1 = (mul_1 - sum_1) - mul_3
    mul_4 = arg3_1 * sub_1
    add = arg4_1 + mul_4  # f32
    ct_1 = add.to(torch.bfloat16)
    ct_2 = arg5_1.to(torch.bfloat16)
    mul_6 = ct_2 * 1.1111111111111112
    mul_7 = ct_1 * mul_6

    # Match strided outputs
    add_strided = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=device, dtype=torch.float32,
    )
    add_strided.copy_(add)

    view_1 = torch.empty_strided(
        flat_shape,
        (flat_shape[1], 1),
        device=device, dtype=torch.bfloat16,
    )
    view_1.copy_(mul_7.view(flat_shape))

    # sum_3 = sum(mul_5, [0,1]) where mul_5 = view * arg2
    mul_5 = view * arg2_1
    mul_5_2d = mul_5.reshape(rows, hidden).contiguous()
    view_2d = view.reshape(rows, hidden).contiguous()

    out_sum3 = torch.empty((BLOCK_C,), device=device, dtype=torch.float32)
    out_sum4 = torch.empty((BLOCK_C,), device=device, dtype=torch.float32)
    out_bf16 = torch.empty((BLOCK_C,), device=device, dtype=torch.float32)

    ROWS_PER_BLOCK = 32
    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _col_sum_kernel,
              (mul_5_2d, out_sum3, rows, hidden, ROWS_PER_BLOCK))
    ct.launch(stream, (1, 1, 1), _col_sum_kernel,
              (view_2d, out_sum4, rows, hidden, ROWS_PER_BLOCK))
    ct.launch(stream, (1, 1, 1), _col_sum_bf16_kernel,
              (view_1, out_bf16, rows, hidden, ROWS_PER_BLOCK))

    sum_3 = torch.empty_strided(sum_shape, (1,), device=device, dtype=torch.float32)
    sum_3.copy_(out_sum3[:hidden])
    sum_4 = torch.empty_strided(sum_shape, (1,), device=device, dtype=torch.float32)
    sum_4.copy_(out_sum4[:hidden])
    out4 = torch.empty_strided(sum_shape, (1,), device=device, dtype=torch.float32)
    out4.copy_(out_bf16[:hidden].to(torch.bfloat16).to(torch.float32))

    return sum_3, sum_4, add_strided, view_1, out4


@oracle_impl(hardware="B200", point="9846b7f2")
def oracle_forward_9846(inputs):
    return _run(inputs, "9846b7f2")


@oracle_impl(hardware="B200", point="5a443972")
def oracle_forward(inputs):
    return _run(inputs, "5a443972")
