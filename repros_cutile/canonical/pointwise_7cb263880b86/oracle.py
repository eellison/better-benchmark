"""cuTile port of pointwise_7cb263880b86: MobileViT QKV right-pad.

Loads Q, K, V from a packed bf16 [B*S, 3*H*HD] projection and writes each into
a (B, H, S, OUT_DIM) output with the last-dim right-zero-padded from HEAD_DIM
to OUT_DIM (= HEAD_DIM + 4). HEAD_DIM is non-power-of-2 (60 or 36), but the
tile is rounded up to a pow2 (64). Padding for the tail uses padding_mode=ZERO
on load, and cuTile silently ignores OOB stores where OUT_DIM < tile width.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _qkv_pad_kernel(
    src5,       # (B, S, 3, H, HD)
    q_out,      # (B, H, S, OUT_DIM)
    k_out,
    v_out,
    HEADS: ct.Constant[int],
    HEAD_DIM_PADDED: ct.Constant[int],
):
    b = ct.bid(0)
    s = ct.bid(1)

    q = ct.load(
        src5,
        index=(b, s, 0, 0, 0),
        shape=(1, 1, 1, HEADS, HEAD_DIM_PADDED),
        padding_mode=ct.PaddingMode.ZERO,
    )
    k = ct.load(
        src5,
        index=(b, s, 1, 0, 0),
        shape=(1, 1, 1, HEADS, HEAD_DIM_PADDED),
        padding_mode=ct.PaddingMode.ZERO,
    )
    v = ct.load(
        src5,
        index=(b, s, 2, 0, 0),
        shape=(1, 1, 1, HEADS, HEAD_DIM_PADDED),
        padding_mode=ct.PaddingMode.ZERO,
    )

    q4 = ct.reshape(q, (1, HEADS, 1, HEAD_DIM_PADDED))
    k4 = ct.reshape(k, (1, HEADS, 1, HEAD_DIM_PADDED))
    v4 = ct.reshape(v, (1, HEADS, 1, HEAD_DIM_PADDED))

    ct.store(q_out, index=(b, 0, s, 0), tile=q4)
    ct.store(k_out, index=(b, 0, s, 0), tile=k4)
    ct.store(v_out, index=(b, 0, s, 0), tile=v4)


@oracle_impl(
    hardware="B200", point="a07ff8de",
    SEQ=16, HEADS=4, HEAD_DIM=60, OUT_DIM=64, HEAD_DIM_PADDED=64,
)
@oracle_impl(
    hardware="B200", point="419b45cb",
    SEQ=256, HEADS=4, HEAD_DIM=36, OUT_DIM=40, HEAD_DIM_PADDED=64,
)
def oracle_forward(inputs, *, SEQ, HEADS, HEAD_DIM, OUT_DIM, HEAD_DIM_PADDED):
    arg0_1 = inputs[0]
    total_rows = int(arg0_1.shape[0])  # B * S
    B = total_rows // SEQ
    src5 = arg0_1.view(B, SEQ, 3, HEADS, HEAD_DIM)

    out_shape = (B, HEADS, SEQ, OUT_DIM)
    out_stride = (HEADS * SEQ * OUT_DIM, SEQ * OUT_DIM, OUT_DIM, 1)
    q_out = torch.empty_strided(
        out_shape, out_stride, device=arg0_1.device, dtype=arg0_1.dtype
    )
    k_out = torch.empty_strided(
        out_shape, out_stride, device=arg0_1.device, dtype=arg0_1.dtype
    )
    v_out = torch.empty_strided(
        out_shape, out_stride, device=arg0_1.device, dtype=arg0_1.dtype
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (B, SEQ, 1),
        _qkv_pad_kernel,
        (src5, q_out, k_out, v_out, HEADS, HEAD_DIM_PADDED),
    )
    return q_out, k_out, v_out
