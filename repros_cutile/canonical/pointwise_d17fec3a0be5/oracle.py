"""cuTile port of pointwise_d17fec3a0be5: MobileBERT affine + residual affine, dual output.

f32 output: (arg0 * arg1 + arg2 + arg3) * arg4 + arg5
bf16 output: bf16(f32_output)
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 32768
COLS = 128


@ct.kernel
def _affine_residual_dual_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    out_f32_ptr,
    out_bf16_ptr,
    N_COLS: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    row_block = ct.bid(0)
    arg0 = ct.astype(
        ct.load(arg0_ptr, index=(row_block, 0), shape=(BLOCK_M, N_COLS)), ct.float32
    )
    arg3 = ct.astype(
        ct.load(arg3_ptr, index=(row_block, 0), shape=(BLOCK_M, N_COLS)), ct.float32
    )
    arg1 = ct.reshape(ct.load(arg1_ptr, index=(0,), shape=(N_COLS,)), (1, N_COLS))
    arg2 = ct.reshape(ct.load(arg2_ptr, index=(0,), shape=(N_COLS,)), (1, N_COLS))
    arg4 = ct.reshape(ct.load(arg4_ptr, index=(0,), shape=(N_COLS,)), (1, N_COLS))
    arg5 = ct.reshape(ct.load(arg5_ptr, index=(0,), shape=(N_COLS,)), (1, N_COLS))

    mul = arg0 * arg1
    add = mul + arg2
    add_1 = arg3 + add
    mul_1 = add_1 * arg4
    add_2 = mul_1 + arg5

    ct.store(out_f32_ptr, index=(row_block, 0), tile=add_2)
    ct.store(out_bf16_ptr, index=(row_block, 0), tile=ct.astype(add_2, ct.bfloat16))


@oracle_impl(hardware="B200", point="8fcb116e", BLOCK_M=64)
def oracle_forward(inputs, *, BLOCK_M: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
        _shape_param_0, _shape_param_1, _shape_param_2,
    ) = inputs
    view_shape = tuple(int(dim) for dim in _shape_param_0)
    view_stride = (view_shape[1] * view_shape[2], view_shape[2], 1)
    out_f32 = torch.empty_strided(
        view_shape,
        view_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out_bf16_base = torch.empty_strided(
        view_shape,
        view_stride,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    # Reshape (256, 128, 128) -> (32768, 128) row-major
    out_f32_flat = out_f32.view(ROWS, COLS)
    out_bf16_flat = out_bf16_base.view(ROWS, COLS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS // BLOCK_M, 1, 1),
        _affine_residual_dual_kernel,
        (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, out_f32_flat, out_bf16_flat, COLS, BLOCK_M),
    )
    return out_f32, out_bf16_base.view(tuple(int(dim) for dim in _shape_param_2))
