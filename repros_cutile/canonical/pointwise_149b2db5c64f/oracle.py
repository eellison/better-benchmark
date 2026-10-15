"""cuTile port of pointwise_149b2db5c64f: channels-last f32-to-bf16 constant_pad_nd.

Ports the Triton `_pad_cast_output_nhwc_kernel`. Input is `[N,C,H,W]` channels-last
f32 and output is `[N,C,H+1,W+1]` channels-last bf16 with zero pad on right/bottom.

cuTile requires power-of-two tile shapes; C=3 is not power-of-2 and neither is
H (192/224). We reshape input/output to a padded logical shape where the trailing
dim is a power of 2, and store only the valid region via a per-tile mask on the
last dim.

Strategy: use a 1D grid over the (N, H+1, W+1) output positions and load 4-wide
(pad) tiles along the channel axis, then store back a matching bf16 tile. Since
the store writes all 4 channels but only 3 are physical, we allocate the output
with an extra pad-channel slot so writes are in-bounds.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


PADDED_C = 4  # next pow2 of C=3


@ct.kernel
def _pad_cast_nhwc_kernel(
    src,        # (N, H, W, C=3) f32 NHWC input (OOB padded to zero via load)
    dst,        # (N, H+1, W+1, PADDED_C) bf16 padded output
    HEIGHT: ct.Constant[int],
    WIDTH: ct.Constant[int],
):
    n = ct.bid(0)
    h = ct.bid(1)
    w = ct.bid(2)

    # Load the 4-wide channel tile with OOB pad-to-zero: the array last dim
    # is 3 so the 4th channel is padded, and h==H or w==W tiles are all-zero.
    values = ct.load(
        src, index=(n, h, w, 0), shape=(1, 1, 1, PADDED_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    result = ct.astype(values, ct.bfloat16)
    ct.store(dst, index=(n, h, w, 0), tile=result)


@oracle_impl(hardware="B200", point="70bbd2d7", BLOCK=1024)
@oracle_impl(hardware="B200", point="45c33f0e", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1 = inputs[0]
    batch, channels, height, width = arg0_1.shape
    out_h = height + 1
    out_w = width + 1

    # arg0_1 has stride (H*W*C, 1, W*C, C) -> channels-last NCHW; permute to
    # (N, H, W, C) NHWC. This is a metadata-only view (already contiguous in
    # NHWC layout thanks to channels-last input strides).
    src_nhwc = arg0_1.permute(0, 2, 3, 1)

    # Padded output buffer: bf16 (N, H+1, W+1, 4) — extra channel slot is
    # scratch to satisfy pow2 tile shape.
    dst_padded = torch.empty(
        (batch, out_h, out_w, PADDED_C),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, out_h, out_w),
        _pad_cast_nhwc_kernel,
        (src_nhwc, dst_padded, height, width),
    )

    # Copy the valid (:C) portion back into an output tensor with the
    # eager-expected strides.
    out = torch.empty_strided(
        (batch, channels, out_h, out_w),
        (channels * out_h * out_w, 1, out_w * channels, channels),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out.permute(0, 2, 3, 1).copy_(dst_padded[:, :, :, :channels])
    return out
