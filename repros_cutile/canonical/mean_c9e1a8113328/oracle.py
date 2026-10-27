"""cuTile port of mean_c9e1a8113328: channels-last spatial mean [N, C, H, W] -> [N, C, 1, 1].

Matches Triton's cl kernel: each program handles (n, c_block) and iterates
over HW in chunks of BLOCK_HW. Reads via NHWC contiguous view (no copy —
channels-last permute is metadata-only for CL inputs).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _spatial_mean_cl_kernel(
    x_ptr,      # bf16 [N, HW, C]  (NHWC contiguous view)
    out_ptr,    # bf16 [N, C]
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    HW_PAD: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    INV_HW: ct.Constant[float],
):
    n = ct.bid(0)
    c_block = ct.bid(1)
    vals = ct.load(x_ptr, index=(n, 0, c_block),
                   shape=(1, HW_PAD, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    vals_f = ct.astype(vals, ct.float32)
    total = ct.sum(vals_f, axis=1)  # (1, BLOCK_C)
    mean = total * INV_HW
    mean_bf = ct.astype(mean, ct.bfloat16)
    ct.store(out_ptr, index=(n, c_block), tile=mean_bf)


@oracle_impl(hardware="B200", point="69db7514", BLOCK_C=256)
@oracle_impl(hardware="B200", point="280fa7d2", BLOCK_C=64)
@oracle_impl(hardware="B200", point="2645a53e", BLOCK_C=64)
@oracle_impl(hardware="B200", point="4ca56f88", BLOCK_C=64)
@oracle_impl(hardware="B200", point="d1686c77", BLOCK_C=256)
@oracle_impl(hardware="B200", point="333b11d0", BLOCK_C=128)
@oracle_impl(hardware="B200", point="547c8164", BLOCK_C=64)
@oracle_impl(hardware="B200", point="f0a667e3", BLOCK_C=32)
@oracle_impl(hardware="B200", point="44f1e75f", BLOCK_C=128)
@oracle_impl(hardware="B200", point="4784273d", BLOCK_C=128)
@oracle_impl(hardware="B200", point="3cf84b69", BLOCK_C=64)
@oracle_impl(hardware="B200", point="2150fa2c", BLOCK_C=64)
@oracle_impl(hardware="B200", point="6b68d184", BLOCK_C=256)
@oracle_impl(hardware="B200", point="d3381898", BLOCK_C=128)
@oracle_impl(hardware="B200", point="473b7062", BLOCK_C=64)
@oracle_impl(hardware="B200", point="630d0797", BLOCK_C=64)
def oracle_forward(inputs, *, BLOCK_C: int):
    (x,) = inputs
    n, c, h, w = (int(dim) for dim in x.shape)
    hw = h * w
    out = torch.empty_strided((n, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.bfloat16)

    # x is channels-last: stride (C*H*W, 1, W*C, C). Permute (0,2,3,1) yields
    # NHWC contiguous — a metadata-only reshape.
    x_nhwc = x.permute(0, 2, 3, 1).reshape(n, hw, c)
    out_2d = out.view(n, c)

    HW_PAD = _next_pow2(hw)
    # Pad BLOCK_C to power of 2 that divides c or use masking.
    BLOCK_C_C = BLOCK_C if (c % BLOCK_C == 0) else _next_pow2(c)
    if c % BLOCK_C_C != 0:
        BLOCK_C_C = _next_pow2(c)

    stream = torch.cuda.current_stream()
    grid = (n, ct.cdiv(c, BLOCK_C_C), 1)
    ct.launch(
        stream,
        grid,
        _spatial_mean_cl_kernel,
        (x_nhwc, out_2d, c, hw, HW_PAD, BLOCK_C_C, 1.0 / hw),
    )
    return out
