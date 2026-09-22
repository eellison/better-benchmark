"""cuTile port of var_mean_f1e6419d4a7f: Swin block window-unshuffle + residual + LN.

The Repro reshapes the flat [25088, 512] into [128, 2, 2, 7, 7, 512] then
permutes to [128, 2, 7, 2, 7, 512] before adding a residual and running
LayerNorm over the last dim. We do the window-unshuffle in torch (metadata
only + one contiguous copy), then run the LN kernel in cuTile.

Outputs: (view_4, view_5) = (add, ln_bf16_flat).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _ln_kernel(
    add_ptr,      # bf16 [rows, HIDDEN]
    weight_ptr,   # bf16 [HIDDEN]
    bias_ptr,     # bf16 [HIDDEN]
    out_ptr,      # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    add_bf = ct.load(add_ptr, index=(row, 0), shape=(1, BLOCK_H),
                     padding_mode=ct.PaddingMode.ZERO)
    x = ct.astype(add_bf, ct.float32)
    inv_h = 1.0 / HIDDEN

    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN, (1, BLOCK_H))
    zero_f = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)
    x_m = ct.where(col_mask, x, zero_f)
    mean = ct.sum(x_m, axis=1, keepdims=True) * inv_h
    centered = x - mean
    centered_m = ct.where(col_mask, centered, zero_f)
    variance = ct.sum(centered_m * centered_m, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    w_f = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H))
    b_f = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H))
    affine = normalized * w_f + b_f
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    known = 1
    missing = -1
    for idx, dim in enumerate(dims):
        if dim == -1:
            missing = idx
        else:
            known *= dim
    if missing >= 0:
        dims[missing] = int(numel) // known
    return tuple(dims)


def _next_pow2(n):
    v = 1
    while v < int(n):
        v <<= 1
    return v


def _run(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3, shape4, shape5 = inputs
    device = arg0_1.device
    numel = arg0_1.numel()

    view0 = arg0_1.view(_resolve_shape(shape0, numel))
    view1 = view0.view(_resolve_shape(shape1, numel))
    view2 = view1.view(_resolve_shape(shape2, numel))
    perm = view2.permute(0, 1, 3, 2, 4, 5).contiguous()
    view3 = perm.view(_resolve_shape(shape3, numel))
    add = (arg1_1.float() + view3.float()).to(torch.bfloat16)
    view4 = add.view(_resolve_shape(shape4, numel))

    rows = view4.shape[0] * view4.shape[1]
    hidden = int(view4.shape[-1])
    device = arg0_1.device
    BLOCK_H = _next_pow2(hidden)

    view5 = torch.empty(_resolve_shape(shape5, numel), device=device, dtype=torch.bfloat16)

    if BLOCK_H == hidden:
        add_2d = view4.view(rows, hidden)
        out_2d = view5.view(rows, hidden)
        weight = arg2_1.view(hidden)
        bias = arg3_1.view(hidden)
    else:
        add_2d = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
        add_2d[:, :hidden].copy_(view4.reshape(rows, hidden))
        out_2d = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
        weight = torch.zeros((BLOCK_H,), device=device, dtype=torch.bfloat16)
        bias = torch.zeros((BLOCK_H,), device=device, dtype=torch.bfloat16)
        weight[:hidden].copy_(arg2_1)
        bias[:hidden].copy_(arg3_1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _ln_kernel,
        (add_2d, weight, bias, out_2d, hidden, BLOCK_H),
    )
    if BLOCK_H != hidden:
        view5.view(rows, hidden).copy_(out_2d[:, :hidden])
    return view4, view5


@oracle_impl(hardware="B200", point="2802cf0f")
@oracle_impl(hardware="B200", point="536b6e86")
@oracle_impl(hardware="B200", point="8abb13ef")
def oracle_forward(inputs):
    return _run(inputs)
