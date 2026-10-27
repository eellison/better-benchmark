"""cuTile port of var_mean_6de4787c0d2d: Swin patch-merge residual LayerNorm.

The Triton oracle fuses the bf16 residual add and the deterministic Swin
patch-merge view/permute/clone into the LayerNorm row kernel. cuTile lacks
inline_asm PTX, but cuTile arithmetic is already RTNE-by-default, so plain
`+`/`*`/`ct.astype(..., ct.bfloat16)` are numerically equivalent.

We rearrange with torch's bf16 add + reshape/permute/contiguous chain to
produce a [rows, hidden] bf16 tile-space input, then run one cuTile row
LayerNorm kernel that mirrors the Triton reduction/affine epilogue.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _row_layernorm_kernel(
    x_ptr,          # bf16 [rows, hidden]
    weight_ptr,     # bf16 [hidden]
    bias_ptr,       # bf16 [hidden]
    out_ptr,        # bf16 [rows, hidden]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    EPS_: ct.Constant[float],
):
    row_block = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    # Match Triton's `residual+flat -> bf16 -> f32` boundary.
    x = ct.astype(ct.astype(x_bf, ct.float32), ct.bfloat16)
    x_f = ct.astype(x, ct.float32)

    inv_h = 1.0 / HIDDEN
    mean_1d = ct.sum(x_f, axis=1, keepdims=True) * inv_h
    centered = x_f - mean_1d
    variance_1d = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    invstd_1d = ct.rsqrt(variance_1d + EPS_)
    normalized = centered * invstd_1d

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    ct.store(out_ptr, index=(row_block, 0), tile=ct.astype(affine, ct.bfloat16))


def _shape(shape):
    return tuple(int(d) for d in shape)


# 6ba63620: BLOCK_H=2048 ROW_BLOCK=1 (hidden=2048)
@oracle_impl(hardware="B200", point="6ba63620", BLOCK_H=2048, ROW_BLOCK=1)
# d431181f: BLOCK_H=1024 ROW_BLOCK=2 (hidden=1024)
@oracle_impl(hardware="B200", point="d431181f", BLOCK_H=1024, ROW_BLOCK=2)
# 2eb9cc7a: BLOCK_H=512  ROW_BLOCK=4 (hidden=512)
@oracle_impl(hardware="B200", point="2eb9cc7a", BLOCK_H=512, ROW_BLOCK=4)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, shape1, _shape2, _shape3, shape4 = inputs
    b = int(shape1[0])
    h = int(shape1[1])
    w = int(shape1[2])
    c = int(shape1[3])
    rows = int(shape4[0])
    hidden = int(shape4[1])
    device = arg0_1.device

    # 1) bf16 residual add matches Triton's `_f32_add(residual, flat).to(bf16)`
    #    — torch bf16 addition is fp32-accumulated then rounded to bf16.
    added = arg1_1 + arg0_1.view(b, h * w, c)                # bf16 [B, H*W, C]
    reshaped = added.view(b, h, w, c)                        # [B, H, W, C]
    part = reshaped.view(b, h // 2, 2, w // 2, 2, c)         # [B, out_h, 2, out_w, 2, C]
    perm = part.permute(0, 1, 3, 4, 2, 5).contiguous()       # [B, out_h, out_w, 2, 2, C]
    merged = perm.view(b, h // 2, w // 2, 4 * c)             # [B, out_h, out_w, 4C]
    flat = merged.view(rows, hidden)                         # [rows, hidden]

    out = torch.empty_strided(
        (rows, hidden), (hidden, 1), device=device, dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _row_layernorm_kernel,
        (flat, arg2_1, arg3_1, out, hidden, BLOCK_H, ROW_BLOCK, EPS),
    )
    return out
