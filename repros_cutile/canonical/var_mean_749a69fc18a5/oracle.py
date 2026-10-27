"""cuTile port of var_mean_749a69fc18a5: GPT-Neo residual LayerNorm forward (hidden=2048)."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _residual_layernorm_kernel(
    bf16_input_ptr,   # bf16 [rows, HIDDEN]
    residual_ptr,     # f32 [rows, HIDDEN]
    weight_ptr,       # f32 [HIDDEN]
    bias_ptr,         # f32 [HIDDEN]
    add_out_ptr,      # f32 [rows, HIDDEN]
    norm_out_ptr,     # f32 [rows, HIDDEN]
    bf16_out_ptr,     # bf16 [rows, HIDDEN]
    invstd_div_ptr,   # f32 [rows]
    HIDDEN: ct.Constant[int],
):
    row = ct.bid(0)
    x0 = ct.load(bf16_input_ptr, index=(row, 0), shape=(1, HIDDEN))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, HIDDEN))
    x = ct.astype(x0, ct.float32) + residual
    ct.store(add_out_ptr, index=(row, 0), tile=x)

    mean = ct.sum(x) * (1.0 / HIDDEN)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)

    normalized = centered * invstd
    ct.store(norm_out_ptr, index=(row, 0), tile=normalized)

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN,))
    bias = ct.load(bias_ptr, index=(0,), shape=(HIDDEN,))
    weight_2d = ct.reshape(weight, (1, HIDDEN))
    bias_2d = ct.reshape(bias, (1, HIDDEN))
    scaled = normalized * weight_2d
    affine = scaled + bias_2d
    ct.store(bf16_out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))
    # invstd is scalar per row; write as size-1 tile
    invstd_scalar = ct.full(shape=(1,), fill_value=0.0, dtype=ct.float32)
    invstd_scalar = ct.reshape(invstd, (1,)) * (1.0 / HIDDEN)
    ct.store(invstd_div_ptr, index=(row,), tile=invstd_scalar)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="5d43e450", BLOCK_H=2048)
def oracle_forward(inputs, *, BLOCK_H: int):
    bf16_input, residual, weight, bias, add_shape, out_shape = inputs
    add_shape = tuple(int(dim) for dim in add_shape)
    out_shape = tuple(int(dim) for dim in out_shape)
    rows = int(bf16_input.shape[0])
    hidden = int(bf16_input.shape[1])

    add_out = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=bf16_input.device,
        dtype=torch.float32,
    )
    norm_out = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=bf16_input.device,
        dtype=torch.float32,
    )
    bf16_out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=bf16_input.device,
        dtype=torch.bfloat16,
    )
    invstd_div = torch.empty_strided(
        (add_shape[0], add_shape[1], 1),
        (add_shape[1], 1, 1),
        device=bf16_input.device,
        dtype=torch.float32,
    )
    # Reshape residual [B, T, H] -> [rows, H]
    residual_2d = residual.reshape(rows, hidden)
    add_out_2d = add_out.view(rows, hidden)
    norm_out_2d = norm_out.view(rows, hidden)
    invstd_1d = invstd_div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_layernorm_kernel,
        (bf16_input, residual_2d, weight, bias, add_out_2d, norm_out_2d, bf16_out, invstd_1d, hidden),
    )
    return add_out, norm_out, bf16_out, invstd_div
