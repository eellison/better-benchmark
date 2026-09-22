"""cuTile port of pointwise_3af9b4ebfd25: MobileBERT residual add + affine.
Emits an f32 add side output plus a bf16 affine output."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_affine_kernel(
    x_ptr,
    residual_ptr,
    scale_ptr,
    bias_ptr,
    add_out_ptr,
    out_ptr,
    N_COLS: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, N_COLS))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, N_COLS))
    scale = ct.load(scale_ptr, index=(0,), shape=(N_COLS,))
    bias = ct.load(bias_ptr, index=(0,), shape=(N_COLS,))

    x_f = ct.astype(x, ct.float32)
    r_f = ct.astype(residual, ct.float32)
    scale_2d = ct.reshape(ct.astype(scale, ct.float32), (1, N_COLS))
    bias_2d = ct.reshape(ct.astype(bias, ct.float32), (1, N_COLS))

    add = x_f + r_f
    y = add * scale_2d + bias_2d
    ct.store(add_out_ptr, index=(row, 0), tile=add)
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(y, ct.bfloat16))


# 4c7d1afa: (T([32768,512], bf16), T([256,128,512], f32), ...)
@oracle_impl(hardware="B200", point="4c7d1afa", N_COLS=512)
# b5045c46: (T([32768,128], bf16), T([256,128,128], f32), ...)
@oracle_impl(hardware="B200", point="b5045c46", N_COLS=128)
def oracle_forward(inputs, *, N_COLS):
    x, residual, scale, bias, _shape0, shape1 = inputs
    rows = int(x.shape[0])
    add_out = torch.empty_strided(
        tuple(residual.shape),
        tuple(residual.stride()),
        device=residual.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        tuple(int(dim) for dim in shape1),
        (N_COLS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    add_flat = add_out.view(rows, N_COLS)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_affine_kernel,
        (x, residual.view(rows, N_COLS), scale, bias, add_flat, out, N_COLS),
    )
    return add_out, out
