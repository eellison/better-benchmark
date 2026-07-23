"""cuTile port of pointwise_f30674e2e5ee: channels-last right-and-bottom zero
`constant_pad_nd`. Copies interior [N,C,H,W] into padded [N,C,H+1,W+1] and
zero-fills the fringe. Runs on the NHWC-flat contiguous views of both tensors
(zero-copy for channels-last). C=3 is padded to BLOCK_C=4 via a scratch buffer.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BLOCK_C = 4  # smallest power of 2 >= C=3


@ct.kernel
def _pad_nhwc_kernel(
    input_ptr,      # bf16 (N, H, W, C)          — cuTile view, contiguous NHWC
    scratch_ptr,    # bf16 (N, out_h, out_w, 4)  — pad C to power of 2
    BLOCK_N: ct.Constant[int],
    BLOCK_C_C: ct.Constant[int],
):
    oh = ct.bid(0)
    ow = ct.bid(1)
    tile = ct.load(
        input_ptr,
        index=(0, oh, ow, 0),
        shape=(BLOCK_N, 1, 1, BLOCK_C_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    ct.store(scratch_ptr, index=(0, oh, ow, 0), tile=tile)


@oracle_impl(hardware="B200", point="33f5cb91", BLOCK=1024)
@oracle_impl(hardware="B200", point="e96942cb", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    del BLOCK  # Triton-only knob
    (x,) = inputs
    N = int(x.shape[0])
    C = int(x.shape[1])
    H = int(x.shape[2])
    W = int(x.shape[3])
    out_h = H + 1
    out_w = W + 1

    assert N & (N - 1) == 0, "BLOCK_N requires power-of-2 N"
    assert C == 3

    # Output with channels-last strides (matches Triton oracle output layout).
    out = torch.empty_strided(
        (N, C, out_h, out_w),
        (C * out_h * out_w, 1, out_w * C, C),
        device=x.device,
        dtype=torch.bfloat16,
    )

    # NHWC-flat views (zero-copy on channels-last inputs).
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    out_nhwc_view = out.permute(0, 2, 3, 1)

    # C=3 -> BLOCK_C=4, so we need a padded scratch to avoid OOB writes.
    scratch = torch.empty(
        (N, out_h, out_w, BLOCK_C),
        device=x.device,
        dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (out_h, out_w, 1), _pad_nhwc_kernel,
        (x_nhwc, scratch, N, BLOCK_C),
    )

    out_nhwc_view.copy_(scratch[..., :C])
    return out
