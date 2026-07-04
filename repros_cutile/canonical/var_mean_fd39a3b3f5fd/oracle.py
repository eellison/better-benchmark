"""cuTile port of var_mean_fd39a3b3f5fd: FNet real-select + residual LayerNorm.

Per-row f32 LayerNorm over hidden=768 (non-pow2 -> split into 3 x 256 tiles).
The real part of arg0 (shape [B, T, H, 2] selected at dim=3 index 0) plus
residual arg1, then correction=0 var_mean, eps=1e-12, affine, and rsqrt/768.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-12
HIDDEN = 768
BLOCK_H = 256  # 3 tiles cover HIDDEN=768


@ct.kernel
def _fnet_layernorm_kernel(
    real_ptr,     # f32 [rows, HIDDEN] (contiguous slice of arg0[..., 0])
    residual_ptr, # f32 [rows, HIDDEN]
    weight_ptr,   # f32 [HIDDEN]
    bias_ptr,     # f32 [HIDDEN]
    normalized_ptr,  # f32 [rows, HIDDEN]
    affine_ptr,      # f32 [rows, HIDDEN]
    invstd_div_ptr,  # f32 [rows]
    HIDDEN_C: ct.Constant[int],
    BLOCK_H_C: ct.Constant[int],
):
    row = ct.bid(0)
    r0 = ct.load(real_ptr, index=(row, 0), shape=(1, BLOCK_H_C))
    r1 = ct.load(real_ptr, index=(row, 1), shape=(1, BLOCK_H_C))
    r2 = ct.load(real_ptr, index=(row, 2), shape=(1, BLOCK_H_C))
    s0 = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H_C))
    s1 = ct.load(residual_ptr, index=(row, 1), shape=(1, BLOCK_H_C))
    s2 = ct.load(residual_ptr, index=(row, 2), shape=(1, BLOCK_H_C))
    x0 = r0 + s0
    x1 = r1 + s1
    x2 = r2 + s2

    mean = (ct.sum(x0) + ct.sum(x1) + ct.sum(x2)) * (1.0 / HIDDEN_C)
    c0 = x0 - mean
    c1 = x1 - mean
    c2 = x2 - mean
    variance = (ct.sum(c0 * c0) + ct.sum(c1 * c1) + ct.sum(c2 * c2)) * (1.0 / HIDDEN_C)
    invstd = ct.rsqrt(variance + EPS)

    n0 = c0 * invstd
    n1 = c1 * invstd
    n2 = c2 * invstd
    ct.store(normalized_ptr, index=(row, 0), tile=n0)
    ct.store(normalized_ptr, index=(row, 1), tile=n1)
    ct.store(normalized_ptr, index=(row, 2), tile=n2)

    w0 = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_C,))
    w1 = ct.load(weight_ptr, index=(1,), shape=(BLOCK_H_C,))
    w2 = ct.load(weight_ptr, index=(2,), shape=(BLOCK_H_C,))
    b0 = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H_C,))
    b1 = ct.load(bias_ptr, index=(1,), shape=(BLOCK_H_C,))
    b2 = ct.load(bias_ptr, index=(2,), shape=(BLOCK_H_C,))
    w0r = ct.reshape(w0, (1, BLOCK_H_C))
    w1r = ct.reshape(w1, (1, BLOCK_H_C))
    w2r = ct.reshape(w2, (1, BLOCK_H_C))
    b0r = ct.reshape(b0, (1, BLOCK_H_C))
    b1r = ct.reshape(b1, (1, BLOCK_H_C))
    b2r = ct.reshape(b2, (1, BLOCK_H_C))

    ct.store(affine_ptr, index=(row, 0), tile=n0 * w0r + b0r)
    ct.store(affine_ptr, index=(row, 1), tile=n1 * w1r + b1r)
    ct.store(affine_ptr, index=(row, 2), tile=n2 * w2r + b2r)

    # Store scalar rsqrt/HIDDEN as size-1 tile
    inv_div = invstd * (1.0 / HIDDEN_C)
    ct.store(invstd_div_ptr, index=(row,), tile=ct.reshape(inv_div, (1,)))


def _shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="98ade792")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0 = inputs
    batch = int(arg1_1.shape[0])
    seq_len = int(arg1_1.shape[1])
    rows = batch * seq_len

    # arg0_1[..., 0] is the real part - select yields a strided [B, T, H] view.
    # cuTile needs the loaded tile to be contiguous, so we materialize via
    # torch.select + contiguous.
    real = arg0_1.select(3, 0).contiguous()  # f32[batch, seq, hidden]

    normalized = torch.empty_like(arg1_1)
    affine = torch.empty_like(arg1_1)
    invstd_div = torch.empty_strided(
        (batch, seq_len, 1),
        (seq_len, 1, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )

    real_2d = real.view(rows, HIDDEN)
    residual_2d = arg1_1.view(rows, HIDDEN)
    normalized_2d = normalized.view(rows, HIDDEN)
    affine_2d = affine.view(rows, HIDDEN)
    invstd_div_flat = invstd_div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _fnet_layernorm_kernel,
        (
            real_2d,
            residual_2d,
            arg2_1,
            arg3_1,
            normalized_2d,
            affine_2d,
            invstd_div_flat,
            HIDDEN,
            BLOCK_H,
        ),
    )
    return normalized, affine, affine.view(_shape(shape0)), invstd_div
