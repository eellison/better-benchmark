"""cuTile port of pointwise_d6d9504b25c0: slice->permute->clone->view materialization.

Loads/stores are split into a 32-wide power-of-two tile plus a 4-wide tail so
cuTile's power-of-two shape constraint is satisfied (input width 40 -> output 36).
Index is tile-space; tail tile of size 4 at src index 8 -> offset 32.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


B = 512
H = 4
T = 256
C_IN = 40
C_OUT = 36


@ct.kernel
def _slice_permute_clone_head_kernel(
    src,  # (B, H, T, C_IN=40) bf16
    dst,  # (B, T, H, C_OUT=36) bf16
):
    batch = ct.bid(0)
    token = ct.bid(1)
    # First 32 channels. src last-dim tile size 32 -> tile-index 0 = elements 0..31.
    # dst last-dim tile size 32 -> tile-index 0 = elements 0..31.
    v = ct.load(src, index=(batch, 0, token, 0), shape=(1, H, 1, 32))
    v = ct.reshape(v, (1, 1, H, 32))
    ct.store(dst, index=(batch, token, 0, 0), tile=v)


@ct.kernel
def _slice_permute_clone_tail_kernel(
    src,  # (B, H, T, C_IN=40)
    dst,  # (B, T, H, C_OUT=36)
):
    batch = ct.bid(0)
    token = ct.bid(1)
    # Next 4 channels: src last-dim tile size 4, tile-index 8 -> elements 32..35.
    # dst last-dim tile size 4, tile-index 8 -> elements 32..35 (of C_OUT=36).
    v = ct.load(src, index=(batch, 0, token, 8), shape=(1, H, 1, 4))
    v = ct.reshape(v, (1, 1, H, 4))
    ct.store(dst, index=(batch, token, 0, 8), tile=v)


@oracle_impl(hardware="B200", point="9bd93817", XBLOCK=1024)
def oracle_forward(inputs, *, XBLOCK):
    x, _shape_param_0, _shape_param_1 = inputs
    out_4d = torch.empty_strided(
        (B, T, H, C_OUT),
        (T * H * C_OUT, H * C_OUT, C_OUT, 1),
        device=x.device,
        dtype=x.dtype,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (B, T, 1),
        _slice_permute_clone_head_kernel,
        (x, out_4d),
    )
    ct.launch(
        stream,
        (B, T, 1),
        _slice_permute_clone_tail_kernel,
        (x, out_4d),
    )
    return out_4d.view(131072, 144)
