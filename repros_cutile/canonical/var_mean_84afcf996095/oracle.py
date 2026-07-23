"""cuTile port of var_mean_84afcf996095: SigLIP ViT patch LayerNorm.

The input arg0_1 has channels-last strides (H*W*C, 1, W*C, C) so we can
view it as (batch, tokens=H*W, hidden=C) via torch.as_strided — this is a
metadata-only reshape, avoiding the .contiguous() copy.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6


@ct.kernel
def _patch_ln_kernel(
    patch_ptr,     # bf16 [rows, HIDDEN]
    pos_ptr,       # bf16 [tokens, HIDDEN]
    weight_ptr,    # bf16 [HIDDEN]
    bias_ptr,      # bf16 [HIDDEN]
    add_out_ptr,   # bf16 [rows, HIDDEN]
    out_ptr,       # bf16 [rows, HIDDEN]
    TOKENS: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    token = row % TOKENS

    patch = ct.load(patch_ptr, index=(row, 0), shape=(1, BLOCK_H))
    pos = ct.load(pos_ptr, index=(token, 0), shape=(1, BLOCK_H))
    added = ct.astype(ct.astype(patch, ct.float32) + ct.astype(pos, ct.float32), ct.bfloat16)
    ct.store(add_out_ptr, index=(row, 0), tile=added)

    x = ct.astype(added, ct.float32)
    mean = ct.sum(x) * (1.0 / HIDDEN)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H))
    bias_2d = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


@oracle_impl(hardware="B200", point="3eee3489", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    view_shape = tuple(int(d) for d in shape0)  # [128, 768, 256]
    out_shape = tuple(int(d) for d in shape1)   # [32768, 768]
    batch = int(view_shape[0])
    hidden = int(view_shape[1])
    tokens = int(view_shape[2])
    rows = batch * tokens

    # arg0_1 has strides [196608, 1, 12288, 768] - channels-last.
    # Physical layout is (batch, H, W, C=hidden) = (batch, tokens, hidden).
    # Use torch.as_strided to view it that way without a copy:
    #   stride: (batch_stride, token_stride, hidden_stride) =
    #           (arg0_1.stride(0), arg0_1.stride(3), arg0_1.stride(1))
    #         = (196608, 768, 1)
    # which matches contiguous [batch, tokens, hidden].
    patch_2d = torch.as_strided(
        arg0_1, (rows, hidden), (hidden, 1))
    pos_2d = arg1_1.view(tokens, hidden)

    add_out = torch.empty_strided(
        (batch, tokens, hidden),
        (tokens * hidden, hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        out_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    add_out_2d = add_out.view(rows, hidden)
    out_2d = out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _patch_ln_kernel,
        (patch_2d, pos_2d, arg2_1, arg3_1, add_out_2d, out_2d, tokens, hidden, BLOCK_H),
    )
    return add_out, out
