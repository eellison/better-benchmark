"""cuTile port of var_mean_f4578c128d74: residual + LayerNorm row kernel.

For each row: added_bf16 = (flat + residual).to(bf16); LayerNorm(added, weight, bias, eps=1e-6).

Uses padding_mode=ZERO on loads and pads outputs to BLOCK_N; when BLOCK_N ==
HIDDEN the output view is a metadata-only reshape; otherwise we do a single
strided copy at the end.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_layernorm_two_output_kernel(
    flat_ptr,       # bf16 [rows, HIDDEN]
    residual_ptr,   # bf16 [rows, HIDDEN]
    weight_ptr,     # bf16 [HIDDEN]
    bias_ptr,       # bf16 [HIDDEN]
    add_out_ptr,    # bf16 [rows, BLOCK_N] (padded)
    norm_out_ptr,   # bf16 [rows, BLOCK_N] (padded)
    HIDDEN: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_N),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_N),
                       padding_mode=ct.PaddingMode.ZERO)
    added_bf16 = ct.astype(ct.astype(flat, ct.float32) + ct.astype(residual, ct.float32),
                           ct.bfloat16)
    ct.store(add_out_ptr, index=(row, 0), tile=added_bf16)

    cols = ct.arange(BLOCK_N, dtype=ct.int32)
    valid = ct.reshape(cols < HIDDEN, (1, BLOCK_N))

    x = ct.astype(added_bf16, ct.float32)
    x_masked = ct.where(valid, x, 0.0)
    mean = ct.sum(x_masked) * (1.0 / HIDDEN)
    centered = x - mean
    centered_masked = ct.where(valid, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + 1.0e-6)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_N,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_N,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_N))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_N))
    y = centered * invstd * weight_2d + bias_2d
    y_bf = ct.astype(y, ct.bfloat16)
    ct.store(norm_out_ptr, index=(row, 0), tile=y_bf)


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    unknown = -1
    known = 1
    for idx, dim in enumerate(dims):
        if dim == -1:
            unknown = idx
        else:
            known *= dim
    if unknown >= 0:
        dims[unknown] = numel // known
    return tuple(dims)


@oracle_impl(hardware="B200", point="155170ab", BLOCK_N=1024)
@oracle_impl(hardware="B200", point="ad6d6241", BLOCK_N=256)
@oracle_impl(hardware="B200", point="a2357153", BLOCK_N=256)
@oracle_impl(hardware="B200", point="f049abfe", BLOCK_N=256)
@oracle_impl(hardware="B200", point="d8c968d2", BLOCK_N=256)
@oracle_impl(hardware="B200", point="9801ab6a", BLOCK_N=1024)
@oracle_impl(hardware="B200", point="c4bf51cc", BLOCK_N=4096)
@oracle_impl(hardware="B200", point="7f824027", BLOCK_N=4096)
@oracle_impl(hardware="B200", point="63bebcf6", BLOCK_N=1024)
@oracle_impl(hardware="B200", point="1cea4d76", BLOCK_N=1024)
@oracle_impl(hardware="B200", point="17affd46", BLOCK_N=1024)
@oracle_impl(hardware="B200", point="0b3dc49f", BLOCK_N=1024)
@oracle_impl(hardware="B200", point="ceab07f0", BLOCK_N=1024)
@oracle_impl(hardware="B200", point="a3e95c29", BLOCK_N=1024)
@oracle_impl(hardware="B200", point="f2e11670", BLOCK_N=1024)
@oracle_impl(hardware="B200", point="324149d9", BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_N):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    add_shape = tuple(int(dim) for dim in _shape_param_0)
    norm_shape = _resolve_shape(_shape_param_1, arg0_1.numel())

    add_out = torch.empty_strided(
        add_shape,
        (add_shape[1] * add_shape[2], add_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        norm_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    residual_2d = arg1_1.view(rows, hidden)
    stream = torch.cuda.current_stream()

    if hidden == BLOCK_N:
        # Fast path: no padding needed, write directly into outputs.
        add_out_2d = add_out.view(rows, hidden)
        ct.launch(
            stream,
            (rows, 1, 1),
            _residual_layernorm_two_output_kernel,
            (arg0_1, residual_2d, arg2_1, arg3_1, add_out_2d, norm_out,
             hidden, BLOCK_N),
        )
    else:
        # Padded internal buffers; single copy back at the end.
        add_padded = torch.empty((rows, BLOCK_N), device=arg0_1.device, dtype=torch.bfloat16)
        norm_padded = torch.empty((rows, BLOCK_N), device=arg0_1.device, dtype=torch.bfloat16)
        ct.launch(
            stream,
            (rows, 1, 1),
            _residual_layernorm_two_output_kernel,
            (arg0_1, residual_2d, arg2_1, arg3_1, add_padded, norm_padded,
             hidden, BLOCK_N),
        )
        add_out.view(rows, hidden).copy_(add_padded[:, :hidden])
        norm_out.copy_(norm_padded[:, :hidden])
    return (add_out, norm_out)
