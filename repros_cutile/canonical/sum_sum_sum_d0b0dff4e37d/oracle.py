"""cuTile port of sum_sum_sum_d0b0dff4e37d: BEiT LayerNorm-backward + projection.

Returns:
  sum_3 = sum(mul_7, [0,1]) where mul_7 = convert_element_type(arg0).f32 * mul_3
  sum_4 = sum(convert_element_type, [0,1])  where convert_element_type = arg0.f32
  add_1 = arg7 + mul_6                      f32
  view_2 = sum(mul_9, [0,1])                where mul_9 = add_1 * view_1.f32
  view_3 = mul_8.bf16                       bf16[25216, 768]
  permute = view_3.T
  out6 = bf16(sum(view_3, dim=0)).f32

Strategy: torch does elementwise; cuTile does column reductions.
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


@oracle_impl(hardware="B200", point="7c1570d5")
def oracle_forward(inputs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
        shape0, shape1, shape2, shape3, shape4,
    ) = inputs
    full_shape = tuple(int(d) for d in shape0)   # [128, 197, 768]
    flat_shape = tuple(int(d) for d in shape3)   # [25216, 768]
    sum_shape_h = tuple(int(d) for d in shape2)  # [768]
    sum_shape_h2 = tuple(int(d) for d in shape4)  # [768]
    device = arg0_1.device
    rows = int(flat_shape[0])
    hidden = int(flat_shape[1])

    # Elementwise per repro.py
    convert_element_type = arg0_1.to(torch.float32).view(full_shape)
    mul = convert_element_type * arg1_1
    mul_1 = mul * 768
    sum_1 = mul.sum(dim=2, keepdim=True)
    view_1 = arg2_1.view(full_shape)  # bf16
    mul_2 = arg3_1 * view_1  # f32 * bf16 → f32
    add = arg4_1 + mul_2
    sub = add - arg5_1
    mul_3 = sub * arg6_1
    mul_4 = mul * mul_3
    sum_2 = mul_4.sum(dim=2, keepdim=True)
    mul_5 = mul_3 * sum_2
    sub_1 = mul_1 - sum_1
    sub_2 = sub_1 - mul_5
    div = arg6_1 / 768
    mul_6 = div * sub_2
    add_1 = arg7_1 + mul_6
    mul_8 = add_1 * arg3_1
    mul_9 = add_1 * view_1  # f32 result (bc bf16 * f32)

    ct_1 = mul_8.to(torch.bfloat16)  # bf16 dense

    add_1_strided = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=device, dtype=torch.float32,
    )
    add_1_strided.copy_(add_1)

    view_3 = torch.empty_strided(
        flat_shape,
        (flat_shape[1], 1),
        device=device, dtype=torch.bfloat16,
    )
    view_3.copy_(ct_1.view(flat_shape))
    permute = view_3.permute(1, 0)

    # Reductions:
    mul_7 = convert_element_type * mul_3  # f32
    mul_7_2d = mul_7.reshape(rows, hidden).contiguous()
    convert_2d = convert_element_type.reshape(rows, hidden).contiguous()
    mul_9_2d = mul_9.reshape(rows, hidden).contiguous()

    out_sum3 = torch.empty((BLOCK_C,), device=device, dtype=torch.float32)
    out_sum4 = torch.empty((BLOCK_C,), device=device, dtype=torch.float32)
    out_sum9 = torch.empty((BLOCK_C,), device=device, dtype=torch.float32)
    out_bf16 = torch.empty((BLOCK_C,), device=device, dtype=torch.float32)

    ROWS_PER_BLOCK = 32
    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _col_sum_kernel,
              (mul_7_2d, out_sum3, rows, hidden, ROWS_PER_BLOCK))
    ct.launch(stream, (1, 1, 1), _col_sum_kernel,
              (convert_2d, out_sum4, rows, hidden, ROWS_PER_BLOCK))
    ct.launch(stream, (1, 1, 1), _col_sum_kernel,
              (mul_9_2d, out_sum9, rows, hidden, ROWS_PER_BLOCK))
    ct.launch(stream, (1, 1, 1), _col_sum_bf16_kernel,
              (view_3, out_bf16, rows, hidden, ROWS_PER_BLOCK))

    sum_3 = torch.empty_strided((hidden,), (1,), device=device, dtype=torch.float32)
    sum_3.copy_(out_sum3[:hidden])
    sum_4 = torch.empty_strided((hidden,), (1,), device=device, dtype=torch.float32)
    sum_4.copy_(out_sum4[:hidden])
    view_2 = torch.empty_strided(sum_shape_h, (1,), device=device, dtype=torch.float32)
    view_2.copy_(out_sum9[:hidden])
    out_final = torch.empty_strided(sum_shape_h2, (1,), device=device, dtype=torch.float32)
    out_final.copy_(out_bf16[:hidden].to(torch.bfloat16).to(torch.float32))

    return sum_3, sum_4, add_1_strided, view_2, view_3, permute, out_final
