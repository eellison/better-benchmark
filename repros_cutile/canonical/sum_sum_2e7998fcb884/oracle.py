"""cuTile port of sum_sum_2e7998fcb884: ALBERT dual-row-sum LayerNorm backward.

One program per row (ROWS=4096); each row sums HIDDEN=4096 in one tile.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8 * 512  # 4096
HIDDEN = 4096


@ct.kernel
def _row_dual_sum_kernel(
    arg0,      # (4096, 4096) bf16
    arg1,      # (8, 512, 4096) f32 -- viewed as (4096, 4096)
    arg2,      # (4096, 4096) bf16
    arg3,      # (4096, 4096) bf16
    arg4,      # (4096,) f32
    arg5,      # (8, 512, 4096) f32 -- viewed as (4096, 4096)
    arg6,      # (8, 512, 1) f32 -- viewed as (4096,)
    add_out,   # (8, 512, 4096) f32 -- viewed as (4096, 4096)
    mul_out,   # (8, 512, 4096) f32 -- viewed as (4096, 4096)
    bf16_out,  # (4096, 4096) bf16
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    base = ct.astype(ct.load(arg1, index=(row, 0), shape=(1, BLOCK_H)), ct.float32)
    a0 = ct.astype(ct.load(arg0, index=(row, 0), shape=(1, BLOCK_H)), ct.float32)
    a2 = ct.astype(ct.load(arg2, index=(row, 0), shape=(1, BLOCK_H)), ct.float32)
    a3 = ct.astype(ct.load(arg3, index=(row, 0), shape=(1, BLOCK_H)), ct.float32)
    add2 = base + a0 + a2 + a3
    ct.store(add_out, index=(row, 0), tile=add2)

    scale = ct.astype(ct.load(arg4, index=(0,), shape=(BLOCK_H,)), ct.float32)
    scale_2d = ct.reshape(scale, (1, BLOCK_H))
    mul = add2 * scale_2d
    rhs = ct.astype(ct.load(arg5, index=(row, 0), shape=(1, BLOCK_H)), ct.float32)
    sum0 = ct.sum(mul, axis=1, keepdims=True)
    sum1 = ct.sum(mul * rhs, axis=1, keepdims=True)

    row_scale_tile = ct.load(arg6, index=(row,), shape=(1,))
    row_scale = ct.reshape(row_scale_tile, (1, 1))
    out = row_scale * ((mul * float(HIDDEN)) - sum0 - (rhs * sum1))
    ct.store(mul_out, index=(row, 0), tile=out)
    ct.store(bf16_out, index=(row, 0), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="b3c0b271", BLOCK_H=HIDDEN)
def oracle_forward(inputs, *, BLOCK_H: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
        _shape_param_0, _shape_param_1, _shape_param_2, shape_param_3,
    ) = inputs

    add_out = torch.empty_like(arg1_1)
    mul_out = torch.empty_like(arg5_1)
    bf16_out = torch.empty_strided(
        tuple(shape_param_3),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    # arg1_1 is (8, 512, 4096) f32; reshape to (ROWS, HIDDEN)
    arg1_2d = arg1_1.view(ROWS, HIDDEN)
    arg5_2d = arg5_1.view(ROWS, HIDDEN)
    arg6_1d = arg6_1.view(ROWS)
    add_out_2d = add_out.view(ROWS, HIDDEN)
    mul_out_2d = mul_out.view(ROWS, HIDDEN)
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _row_dual_sum_kernel,
        (
            arg0_1, arg1_2d, arg2_1, arg3_1, arg4_1, arg5_2d, arg6_1d,
            add_out_2d, mul_out_2d, bf16_out, BLOCK_H,
        ),
    )
    return add_out, mul_out, bf16_out, bf16_out.permute(1, 0)
