"""cuTile port of var_mean_96edf0542ecd: residual-add LayerNorm row kernel."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_layernorm_view_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_N))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_N))
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_N,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_N,))

    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    add_bf16 = ct.astype(flat_f + resid_f, ct.bfloat16)
    x = ct.astype(add_bf16, ct.float32)

    mean = ct.sum(x) * (1.0 / HIDDEN)
    centered = x - mean
    var = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(var + 1.0e-12)

    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_N))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_N))
    y = ct.astype(centered * invstd * weight_2d + bias_2d, ct.bfloat16)

    ct.store(out_ptr, index=(row, 0), tile=y)


@oracle_impl(hardware="B200", point="d4701d13", BLOCK_N=4096)
@oracle_impl(hardware="B200", point="63bebcf6", BLOCK_N=1024)
@oracle_impl(hardware="B200", point="107053b2", BLOCK_N=2048)
@oracle_impl(hardware="B200", point="82b81ff3", BLOCK_N=1024)
@oracle_impl(hardware="B200", point="94a8a62c", BLOCK_N=256)
def oracle_forward(inputs, *, BLOCK_N):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    out_shape = tuple(int(dim) for dim in _shape_param_1)

    out = torch.empty_strided(
        out_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    # arg1_1 is [B, S, H] contiguous; view flat as [rows, H] to match arg0_1
    residual_2d = arg1_1.view(rows, hidden)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_layernorm_view_kernel,
        (arg0_1, residual_2d, arg2_1, arg3_1, out, hidden, BLOCK_N),
    )
    return out
