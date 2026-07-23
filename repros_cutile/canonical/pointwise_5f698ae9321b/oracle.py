"""cuTile port of pointwise_5f698ae9321b: MobileBERT bf16 residual affine.

NEW_PATTERN: for a bf16 flat [32768, 128] input and bf16 [256, 128, 128]
residual (same storage), scale/bias per-feature vectors of length 128, compute
add(bf16 -> f32) + residual(f32) with bf16 rounding, scale in f32 with bf16
rounding, bias in f32 with bf16 rounding. Return the [256, 128, 128] output
and its flat [32768, 128] alias view.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_residual_affine_kernel(
    flat_ptr,      # bf16 [N_ROWS, N_COLS]
    residual_ptr,  # bf16 [N_ROWS, N_COLS]
    scale_ptr,     # bf16 [N_COLS]
    bias_ptr,      # bf16 [N_COLS]
    out_ptr,       # bf16 [N_ROWS, N_COLS]
    N_COLS: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    residual = ct.load(residual_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    scale = ct.load(scale_ptr, index=(0,), shape=(BLOCK_N,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_N,))

    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    scale_f = ct.astype(scale, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    scale_2d = ct.reshape(scale_f, (1, BLOCK_N))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_N))

    add_value = ct.astype(ct.astype(flat_f + resid_f, ct.bfloat16), ct.float32)
    mul_value = ct.astype(ct.astype(add_value * scale_2d, ct.bfloat16), ct.float32)
    value = ct.astype(mul_value + bias_2d, ct.bfloat16)
    ct.store(out_ptr, index=(row_block, 0), tile=value)


@oracle_impl(hardware="B200", point="d2ddc3c7", BLOCK_M=8, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    cols = int(arg0_1.shape[1])
    view_shape = tuple(int(dim) for dim in _shape_param_0)
    final_shape = tuple(int(dim) for dim in _shape_param_1)

    output = torch.empty_strided(
        view_shape,
        (view_shape[1] * view_shape[2], view_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    output_flat = output.view(rows, cols)
    residual_flat = arg1_1.view(rows, cols)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        ((rows + BLOCK_M - 1) // BLOCK_M, 1, 1),
        _bf16_residual_affine_kernel,
        (arg0_1, residual_flat, arg2_1, arg3_1, output_flat, cols, BLOCK_M, BLOCK_N),
    )
    return output, output.view(final_shape)
