"""cuTile port of sum_sum_sum_127fc8edd5da: DistilBERT LayerNorm-backward tail.

Hybrid strategy:
  * Compute all elementwise ops (add, mul, mul_4=dense_f32) via torch to match
    eager's exact fp32 op ordering (which uses bit-exact rtne through fusion).
  * Use cuTile kernels for the column reductions [768] over rows [32768].

Reference (from repro.py):
  add   = arg1 + arg0.f32       f32[256, 128, 768]
  mul   = add * arg2            f32
  mul_2 = mul * arg3            f32
  sum_1 = sum(mul, dim=-1)      f32[..., 1]
  sum_2 = sum(mul_2, dim=-1)    f32
  mul_3 = arg3 * sum_2
  sub_1 = mul*768 - sum_1 - mul_3
  mul_4 = arg4 * sub_1          f32
  dense_bf16 = mul_4.bf16
  sum_3 = sum(add * arg3, [0,1])       f32[768]
  sum_4 = sum(add, [0,1])              f32[768]
  view_1 = dense_bf16.view(32768, 768)
  sum_5 = view_1.sum(dim=0, dtype=f32) f32[768]
  out5 = sum_5.bf16.f32                f32[768]
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 768
BLOCK_C = 1024   # padded pow2


@ct.kernel
def _col_sum_kernel(
    x_ptr,          # f32 [ROWS, HIDDEN]
    out_ptr,        # f32 [HIDDEN]
    ROWS_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    ROWS_PER_BLOCK: ct.Constant[int],
):
    """Sum along dim=0 (rows), writes [HIDDEN]."""
    pid = ct.bid(0)   # one program handles column tile at (pid,)
    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    col_mask = cols < HIDDEN_

    acc = ct.zeros((BLOCK_C,), dtype=ct.float32)

    num_row_tiles = ct.cdiv(ROWS_, ROWS_PER_BLOCK)
    for row_tile in range(num_row_tiles):
        row_start = row_tile * ROWS_PER_BLOCK
        for local_row in ct.static_iter(range(ROWS_PER_BLOCK)):
            row = row_start + local_row
            row_active = row < ROWS_
            if row_active:
                x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_C),
                             padding_mode=ct.PaddingMode.ZERO)
                x_1d = ct.reshape(x, (BLOCK_C,))
                acc = acc + ct.where(col_mask, x_1d, 0.0)

    ct.store(out_ptr, index=(0,), tile=acc)


@ct.kernel
def _col_sum_bf16_kernel(
    x_ptr,          # bf16 [ROWS, HIDDEN]
    out_ptr,        # f32 [HIDDEN]
    ROWS_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    ROWS_PER_BLOCK: ct.Constant[int],
):
    """Sum bf16 tensor along dim=0 (rows) using fp32 accumulator."""
    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    col_mask = cols < HIDDEN_

    acc = ct.zeros((BLOCK_C,), dtype=ct.float32)

    num_row_tiles = ct.cdiv(ROWS_, ROWS_PER_BLOCK)
    for row_tile in range(num_row_tiles):
        row_start = row_tile * ROWS_PER_BLOCK
        for local_row in ct.static_iter(range(ROWS_PER_BLOCK)):
            row = row_start + local_row
            row_active = row < ROWS_
            if row_active:
                x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_C),
                             padding_mode=ct.PaddingMode.ZERO)
                x_f = ct.astype(x, ct.float32)
                x_1d = ct.reshape(x_f, (BLOCK_C,))
                acc = acc + ct.where(col_mask, x_1d, 0.0)

    ct.store(out_ptr, index=(0,), tile=acc)


@oracle_impl(hardware="B200", point="953b9872")
def oracle_forward(inputs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1,
        shape0, shape1, shape2,
    ) = inputs

    full_shape = tuple(int(d) for d in shape0)  # [256,128,768]
    flat_shape = tuple(int(d) for d in shape1)  # [32768,768]
    sum_shape = tuple(int(d) for d in shape2)   # [768]
    device = arg0_1.device
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])

    # Match eager op sequence exactly using torch (goes through same rtne kernels)
    view = arg0_1.view(full_shape)
    add = arg1_1 + view.to(torch.float32)
    mul = add * arg2_1
    mul_1 = mul * 768
    sum_1 = mul.sum(dim=2, keepdim=True)
    mul_2 = mul * arg3_1
    sum_2 = mul_2.sum(dim=2, keepdim=True)
    mul_3 = arg3_1 * sum_2
    sub = mul_1 - sum_1
    sub_1 = sub - mul_3
    mul_4 = arg4_1 * sub_1                         # f32 dense
    dense_bf16 = mul_4.to(torch.bfloat16)          # bf16 dense

    # Reshape to expected strided output for mul_4
    mul_4_strided = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=device, dtype=torch.float32,
    )
    mul_4_strided.copy_(mul_4)

    view_1 = torch.empty_strided(
        flat_shape,
        (flat_shape[1], 1),
        device=device, dtype=torch.bfloat16,
    )
    view_1.copy_(dense_bf16.view(flat_shape))

    permute = view_1.permute(1, 0)

    # sum_3 = sum(add * arg3, [0,1])
    mul_5 = add * arg3_1                 # f32
    mul_5_2d = mul_5.view(rows, hidden)
    add_2d = add.view(rows, hidden)

    # Column reductions using cuTile kernels
    out_x_rhs = torch.empty((BLOCK_C,), device=device, dtype=torch.float32)
    out_x = torch.empty((BLOCK_C,), device=device, dtype=torch.float32)
    out_bf16 = torch.empty((BLOCK_C,), device=device, dtype=torch.float32)

    ROWS_PER_BLOCK = 32
    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _col_sum_kernel,
              (mul_5_2d.contiguous(), out_x_rhs, rows, hidden, ROWS_PER_BLOCK))
    ct.launch(stream, (1, 1, 1), _col_sum_kernel,
              (add_2d.contiguous(), out_x, rows, hidden, ROWS_PER_BLOCK))
    ct.launch(stream, (1, 1, 1), _col_sum_bf16_kernel,
              (view_1, out_bf16, rows, hidden, ROWS_PER_BLOCK))

    sum_3 = torch.empty_strided(sum_shape, (1,), device=device, dtype=torch.float32)
    sum_3.copy_(out_x_rhs[:hidden])
    sum_4 = torch.empty_strided(sum_shape, (1,), device=device, dtype=torch.float32)
    sum_4.copy_(out_x[:hidden])
    # out5 = bf16(sum_5).f32
    out5 = torch.empty_strided(sum_shape, (1,), device=device, dtype=torch.float32)
    out5.copy_(out_bf16[:hidden].to(torch.bfloat16).to(torch.float32))

    return mul_4_strided, sum_3, sum_4, view_1, permute, out5
