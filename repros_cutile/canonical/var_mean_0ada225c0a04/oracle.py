"""cuTile port of var_mean_0ada225c0a04: residual-add LayerNorm row kernel.

Loads a full (ROW_BLOCK, BLOCK_H) tile; row var_mean; affine epilogue; bf16 out.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_layernorm_kernel(
    flat,      # (ROWS, HIDDEN) bf16
    residual,  # (ROWS, HIDDEN) bf16
    weight,    # (HIDDEN,) bf16
    bias,      # (HIDDEN,) bf16
    out,       # (ROWS, HIDDEN) bf16
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    row_block = ct.bid(0)
    f = ct.load(flat, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    r = ct.load(residual, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    x = ct.astype(
        ct.astype(ct.astype(r, ct.float32) + ct.astype(f, ct.float32), ct.bfloat16),
        ct.float32,
    )

    mean = ct.sum(x, axis=1, keepdims=True) * (1.0 / HIDDEN)
    centered = x - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + 1.0e-6)

    w = ct.astype(ct.load(weight, index=(0,), shape=(BLOCK_H,)), ct.float32)
    b = ct.astype(ct.load(bias, index=(0,), shape=(BLOCK_H,)), ct.float32)
    w_2d = ct.reshape(w, (1, BLOCK_H))
    b_2d = ct.reshape(b, (1, BLOCK_H))
    normalized = centered * invstd
    affine = normalized * w_2d + b_2d
    ct.store(out, index=(row_block, 0), tile=ct.astype(affine, ct.bfloat16))


def _launch(inputs, *, BLOCK_H, ROW_BLOCK):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    out_shape = tuple(int(d) for d in _shape_param_1)
    out = torch.empty_strided(out_shape, (hidden, 1), device=arg0_1.device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    # out and inputs 2D view of shape (rows, hidden). arg1_1 may be 3D.
    out_2d = out.view(rows, hidden)
    flat_2d = arg0_1.view(rows, hidden)
    resid_2d = arg1_1.reshape(rows, hidden)
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _residual_layernorm_kernel,
        (flat_2d, resid_2d, arg2_1, arg3_1, out_2d, hidden, BLOCK_H, ROW_BLOCK),
    )
    return out


@oracle_impl(hardware="B200", point="c4bf51cc", BLOCK_H=4096, ROW_BLOCK=1, USE_EXACT=False)
@oracle_impl(hardware="B200", point="7f824027", BLOCK_H=4096, ROW_BLOCK=1, USE_EXACT=False)
@oracle_impl(hardware="B200", point="f2c837cd", BLOCK_H=2048, ROW_BLOCK=1, USE_EXACT=False)
@oracle_impl(hardware="B200", point="17affd46", BLOCK_H=1024, ROW_BLOCK=2, USE_EXACT=False)
@oracle_impl(hardware="B200", point="0b3dc49f", BLOCK_H=1024, ROW_BLOCK=2, USE_EXACT=False)
@oracle_impl(hardware="B200", point="ceab07f0", BLOCK_H=1024, ROW_BLOCK=2, USE_EXACT=False)
@oracle_impl(hardware="B200", point="a3e95c29", BLOCK_H=1024, ROW_BLOCK=2, USE_EXACT=False)
@oracle_impl(hardware="B200", point="d4cc3e3e", BLOCK_H=1024, ROW_BLOCK=2, USE_EXACT=False)
@oracle_impl(hardware="B200", point="f2e11670", BLOCK_H=1024, ROW_BLOCK=2, USE_EXACT=False)
@oracle_impl(hardware="B200", point="9801ab6a", BLOCK_H=1024, ROW_BLOCK=4, USE_EXACT=True)
@oracle_impl(hardware="B200", point="1cea4d76", BLOCK_H=1024, ROW_BLOCK=4, USE_EXACT=True)
@oracle_impl(hardware="B200", point="d1f40ce2", BLOCK_H=1024, ROW_BLOCK=4, USE_EXACT=False)
@oracle_impl(hardware="B200", point="324149d9", BLOCK_H=512, ROW_BLOCK=1, USE_EXACT=False)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK, USE_EXACT):
    return _launch(inputs, BLOCK_H=BLOCK_H, ROW_BLOCK=ROW_BLOCK)
