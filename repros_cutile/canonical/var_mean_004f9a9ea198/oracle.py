"""cuTile port of var_mean_004f9a9ea198 (SCHEDULER_FUSION): Swin window-reverse
cyclic-shift residual LayerNorm. Materialize the reverse-and-shift residual add
via torch, then apply the fixed-hidden LayerNorm via cuTile.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 256


@ct.kernel
def _layernorm_row_kernel(
    x_ptr,           # bf16 [ROWS, HIDDEN]
    weight_ptr,      # bf16 [HIDDEN]
    bias_ptr,        # bf16 [HIDDEN]
    out_ptr,         # bf16 [ROWS, HIDDEN]
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK_: ct.Constant[int],
):
    row_block = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(row_block, 0), shape=(ROW_BLOCK_, BLOCK_H))
    x = ct.astype(x_bf, ct.float32)
    inv_h = 1.0 / HIDDEN_
    x_sum = ct.sum(x, axis=1, keepdims=True)
    x_sq_sum = ct.sum(x * x, axis=1, keepdims=True)
    mean = x_sum * inv_h
    variance = x_sq_sum * inv_h - mean * mean
    invstd = ct.rsqrt(variance + 1.0e-5)

    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    centered = x - mean
    affine = centered * invstd * weight_2d + bias_2d
    ct.store(out_ptr, index=(row_block, 0), tile=ct.astype(affine, ct.bfloat16))


@oracle_impl(hardware="B200", point="536b6e86", BLOCK_H=256, ROW_BLOCK=8)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK):
    arg0_1, arg1_1, arg2_1, arg3_1, _s0, _s1, _s2, _s3, _s4, _s5 = inputs
    view4_shape = tuple(int(dim) for dim in _s4)  # may contain -1
    view5_shape = tuple(int(dim) for dim in _s5)

    # Materialize view_4 = residual + shifted_reversed(window)
    view0 = arg0_1.view(tuple(int(dim) for dim in _s0))
    view1 = view0.view(tuple(int(dim) for dim in _s1))
    view2 = view1.view(tuple(int(dim) for dim in _s2))
    permute = view2.permute(0, 1, 3, 2, 4, 5).contiguous()
    view3 = permute.view(tuple(int(dim) for dim in _s3))
    # Cyclic shift by 25 in dims 1 and 2.
    idx = (torch.arange(28, device=arg0_1.device) + 25) % 28
    shifted = view3[:, idx][:, :, idx]
    add_val = arg1_1 + shifted
    view4 = add_val.view(view4_shape)

    # LayerNorm row-wise.
    hidden = int(arg2_1.shape[0])
    rows = view4.numel() // hidden
    x_2d = view4.reshape(rows, hidden)

    view5 = torch.empty(view5_shape, device=arg0_1.device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _layernorm_row_kernel,
        (x_2d, arg2_1, arg3_1, view5, hidden, BLOCK_H, ROW_BLOCK),
    )
    return view4, view5
