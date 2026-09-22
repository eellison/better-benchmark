"""cuTile port of mean_b2700cd0ebd1: NFNet SiLU + spatial mean (channels-last).

Input x is bf16 [N, C, H, W] in channels-last layout. Compute per-(n,c):
  silu(x) = x / (exp(-x) + 1),  rounded to bf16 -> back to f32,
  mean over (H, W) -> bf16 output [N, C].

We iterate one (n, c) at a time; permute the input to [N, H, W, C] (contiguous
in channels-last format) so a single load picks up the C-th channel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _silu_spatial_mean_kernel(
    x_nhwc_ptr,   # bf16 [N, HW, C]  (channels-last reshaped)
    out_ptr,      # bf16 [N, C]
    HW: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    tile = ct.load(x_nhwc_ptr, index=(n, 0, c), shape=(1, BLOCK_HW, 1),
                   padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(tile, ct.float32)
    silu = x_f / (ct.exp(-x_f) + 1.0)
    rounded = ct.astype(ct.astype(silu, ct.bfloat16), ct.float32)
    total = ct.sum(rounded)
    mean = total * (1.0 / HW)
    ct.store(out_ptr, index=(n, c), tile=ct.astype(mean, ct.bfloat16))


@oracle_impl(hardware="B200", point="ec934f37")
@oracle_impl(hardware="B200", point="9cb825ed")
def oracle_forward(inputs):
    x, _sp0, _sp1, out_shape_arg = inputs
    n, c, h, w = (int(d) for d in x.shape)
    hw = h * w
    # Input has channels-last strides (N, C, H, W) with stride (H*W*C, 1, W*C, C).
    # After permute(0, 2, 3, 1) the tensor is already contiguous in [N, H, W, C].
    x_nhwc_flat = x.permute(0, 2, 3, 1).reshape(n, hw, c)
    # Ensure it's a proper 4D torch view.
    out_shape = tuple(int(d) for d in out_shape_arg)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    # Pick BLOCK_HW = next pow2 >= hw.
    block_hw = 1
    while block_hw < hw:
        block_hw *= 2
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n, c, 1),
        _silu_spatial_mean_kernel,
        (x_nhwc_flat, out, hw, block_hw),
    )
    return out
