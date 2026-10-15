"""cuTile port of pointwise_288dc69505cc: NFNet exact-erf GELU + right/bottom pad.

Pre-compute erf outside the kernel via torch.special.erf (cuTile has no erf).
A cuTile kernel then does the GELU*scale pointwise and writes a bf16 tensor,
which is copied into the interior of a zero-initialized padded output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 1.7015043497085571


@ct.kernel
def _gelu_scale_kernel(
    x_ptr,      # bf16 [total] (input, contiguous)
    erf_ptr,    # bf16 [total] (precomputed erf(x*0.7071..))
    out_ptr,    # bf16 [total] (interior output)
    N: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO)
    e = ct.load(erf_ptr, index=(pid,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)
    e_f = ct.astype(e, ct.float32)
    half = x_f * 0.5
    gelu = half * (e_f + 1.0)
    gelu_bf16 = ct.astype(gelu, ct.bfloat16)
    scaled = ct.astype(gelu_bf16, ct.float32) * SCALE
    scaled_bf16 = ct.astype(scaled, ct.bfloat16)

    # Mask OOB before store using scatter.
    idx = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    valid = idx < N
    ct.scatter(out_ptr, (idx,), scaled_bf16, mask=valid)


def _launch(inputs):
    arg0_1 = inputs[0]
    batch, channels, height, width = arg0_1.shape
    out_h = height + 1
    out_w = width + 1

    out = torch.empty_strided(
        (batch, channels, out_h, out_w),
        (channels * out_h * out_w, 1, out_w * channels, channels),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out.zero_()

    # Precompute erf(x * 0.7071...) outside the kernel (cuTile lacks erf).
    x_f32 = arg0_1.to(torch.float32)
    erf_val = torch.special.erf(x_f32 * 0.7071067811865476).to(torch.bfloat16)

    # Compute GELU * scale into a contiguous interior buffer using NHWC layout.
    # Reshape input/erf to NHWC contiguous 1D for pointwise ops.
    x_nhwc = arg0_1.permute(0, 2, 3, 1).contiguous()
    erf_nhwc = erf_val.permute(0, 2, 3, 1).contiguous()
    interior_nhwc = torch.empty(
        (batch, height, width, channels),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    total = interior_nhwc.numel()
    BLOCK = 1024
    grid = ((total + BLOCK - 1) // BLOCK, 1, 1)
    x_flat = x_nhwc.view(-1)
    erf_flat = erf_nhwc.view(-1)
    interior_flat = interior_nhwc.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        grid,
        _gelu_scale_kernel,
        (x_flat, erf_flat, interior_flat, total, BLOCK),
    )

    # Copy interior into the interior region of the zero-padded output.
    # interior_nhwc is [N, H, W, C]; permute back to [N, C, H, W].
    interior_nchw = interior_nhwc.permute(0, 3, 1, 2)
    out[:, :, :height, :width].copy_(interior_nchw)
    return out


@oracle_impl(hardware="B200", point="192c8e99")
@oracle_impl(hardware="B200", point="94a90df4")
@oracle_impl(hardware="B200", point="4ca56f88")
@oracle_impl(hardware="B200", point="a776dd34")
@oracle_impl(hardware="B200", point="95428590")
@oracle_impl(hardware="B200", point="7b76ed5c")
@oracle_impl(hardware="B200", point="f0a667e3")
@oracle_impl(hardware="B200", point="c5d259c9")
def oracle_forward(inputs):
    return _launch(inputs)
