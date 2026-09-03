"""cuTile port of pointwise_02974718a634: residual affine (three outputs)."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_affine_three_output_kernel(
    flat_bf16,  # (rows, N_COLS) bf16
    residual,   # (rows, N_COLS) f32
    scale,      # (N_COLS,) f32
    bias,       # (N_COLS,) f32
    add_out,    # (rows, N_COLS) f32
    affine_out, # (rows, N_COLS) f32
    bf16_out,   # (rows, N_COLS) bf16
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)

    flat = ct.load(flat_bf16, index=(row_block, col_block), shape=(BLOCK_M, BLOCK_N))
    res = ct.load(residual, index=(row_block, col_block), shape=(BLOCK_M, BLOCK_N))
    s = ct.load(scale, index=(col_block,), shape=(BLOCK_N,))
    b = ct.load(bias, index=(col_block,), shape=(BLOCK_N,))

    flat_f = ct.astype(flat, ct.float32)
    add_v = flat_f + res
    s_2d = ct.reshape(s, (1, BLOCK_N))
    b_2d = ct.reshape(b, (1, BLOCK_N))
    affine_v = add_v * s_2d + b_2d
    bf16_v = ct.astype(affine_v, ct.bfloat16)

    ct.store(add_out, index=(row_block, col_block), tile=add_v)
    ct.store(affine_out, index=(row_block, col_block), tile=affine_v)
    ct.store(bf16_out, index=(row_block, col_block), tile=bf16_v)


@oracle_impl(hardware="B200", point="b5045c46", BLOCK_M=8, BLOCK_N=128)
@oracle_impl(hardware="B200", point="4c7d1afa", BLOCK_M=4, BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    cols = int(arg0_1.shape[1])
    view_shape = tuple(int(dim) for dim in _shape_param_0)
    final_shape = tuple(int(dim) for dim in _shape_param_1)

    add_out = torch.empty_strided(
        view_shape,
        (view_shape[1] * view_shape[2], view_shape[2], 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    affine_out = torch.empty_strided(
        view_shape,
        (view_shape[1] * view_shape[2], view_shape[2], 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    bf16_out = torch.empty_strided(
        final_shape,
        (final_shape[1], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    # Reshape residual/affine_out/add_out from (256,128,128) to (rows, cols)
    residual_2d = arg1_1.view(rows, cols)
    add_2d = add_out.view(rows, cols)
    affine_2d = affine_out.view(rows, cols)

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(rows, BLOCK_M), ct.cdiv(cols, BLOCK_N), 1)
    ct.launch(
        stream,
        grid,
        _residual_affine_three_output_kernel,
        (arg0_1, residual_2d, arg2_1, arg3_1, add_2d, affine_2d, bf16_out, BLOCK_M, BLOCK_N),
    )
    return add_out, affine_out, bf16_out
