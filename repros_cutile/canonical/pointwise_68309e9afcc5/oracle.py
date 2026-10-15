"""cuTile port of pointwise_68309e9afcc5: MobileViT BN + SiLU + channel-cat.

For each output element index i (flat, in the channels-last output layout
`(batch, out_c, height, width)` with strides `(oc*H*W, 1, oc*W, oc)` so
elements are ordered by (b, h, w, out_c)):
- If out_c < C: copy residual[b, out_c, h, w]  (from `arg5_1`, same channels-last).
- If out_c >= C: compute BN(arg1_1[b, out_c-C, h, w]) -> SiLU -> bf16 -> write.

We split into two kernels: one writes the residual half, the other writes
the BN+SiLU half.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _residual_copy_kernel(
    residual_ptr,  # bf16 channels-last [b, c, h, w]
    out_ptr,       # bf16 channels-last [b, 2c, h, w]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(residual_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(out_ptr, index=(pid,), tile=x)


@ct.kernel
def _bn_silu_kernel(
    x_ptr,         # bf16 [b, c, h, w] channels-last flat (element order (b,h,w,c))
    mean_ptr,      # bf16 [c]
    var_ptr,       # bf16 [c]
    weight_ptr,    # bf16 [c]
    bias_ptr,      # bf16 [c]
    out_ptr,       # bf16 flat [b*c*h*w]
    C: ct.Constant[int],
    BLOCK: ct.Constant[int],
    EPS_C: ct.Constant[float],
):
    pid = ct.bid(0)
    lane = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    # channel index within the tile: element at lane `l` (in flat layout with
    # channels innermost) has c = l % C.
    c_idx = lane % C

    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    x_f = ct.astype(x, ct.float32)

    mean = ct.gather(mean_ptr, c_idx)
    var = ct.gather(var_ptr, c_idx)
    weight = ct.gather(weight_ptr, c_idx)
    bias = ct.gather(bias_ptr, c_idx)
    mean_f = ct.astype(mean, ct.float32)
    var_f = ct.astype(var, ct.float32)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)

    invstd = 1.0 / ct.sqrt(var_f + EPS_C)
    affine = (x_f - mean_f) * invstd * weight_f + bias_f
    rounded = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    silu = rounded / (ct.exp(-rounded) + 1.0)
    silu_bf = ct.astype(silu, ct.bfloat16)

    ct.store(out_ptr, index=(pid,), tile=silu_bf)


def _launch(inputs, *, BLOCK: int):
    mean, x, var, weight, bias, residual = inputs
    batch, channels, height, width = x.shape
    hw = height * width
    out_channels = channels * 2

    # Output has channels-last strides (oc*H*W, 1, oc*W, oc).
    out = torch.empty_strided(
        (batch, out_channels, height, width),
        (out_channels * hw, 1, out_channels * width, out_channels),
        device=x.device,
        dtype=torch.bfloat16,
    )

    # Flat layout is (b, h, w, out_c). Split at out_c==C:
    # For each (b, h, w), the first C slots are residual and the next C
    # are BN(x). But it's cleaner to iterate over the flat index and let
    # the kernel branch on c_idx>=C — actually splitting into two kernels
    # over separate output "slabs" is simpler if the output backing memory
    # is written directly. But the memory is interleaved!
    #
    # Trick: as_strided out into two [batch, C, height, width] views with
    # channels-last strides but starting offsets 0 (first C) and C (second C).
    # out.as_strided(shape=(B,C,H,W), stride=(2C*H*W, 1, 2C*W, 2C), storage_offset=0)
    # writes the first C channels;
    # out.as_strided(..., storage_offset=C) writes the second C channels.
    residual_view = out.as_strided(
        (batch, channels, height, width),
        (out_channels * hw, 1, out_channels * width, out_channels),
        storage_offset=0,
    )
    bn_view = out.as_strided(
        (batch, channels, height, width),
        (out_channels * hw, 1, out_channels * width, out_channels),
        storage_offset=channels,
    )

    # Now the RESIDUAL view has the SAME strides as the input `residual` tensor
    # (which is bf16 channels-last with strides (C*H*W, 1, C*W, C)).
    # Wait — residual has strides (C*H*W, 1, C*W, C) but out is (2C*H*W, 1, 2C*W, 2C).
    # NOT the same! Elements written into `residual_view` land at positions
    # spaced by 2C. We can't just do a memcpy-like kernel with contiguous BLOCK
    # since the elements are strided.
    #
    # Simplest correct approach: do a torch permute / native copy for the
    # residual write, and a kernel for the BN+SiLU + write.

    # Residual: `residual` is [B, C, H, W] channels-last. Write into residual_view.
    # This is a strided copy that torch handles well.
    residual_view.copy_(residual)

    # BN+SiLU: compute into a contiguous scratch, then strided-copy into bn_view.
    scratch = torch.empty(
        (batch, channels, height, width),
        device=x.device,
        dtype=torch.bfloat16,
    )
    # We'll produce scratch in NHWC order (i.e. flat layout with c innermost).
    # Then copy into bn_view which has NHWC strides — that copy is native.
    x_perm = x.permute(0, 2, 3, 1).contiguous().view(-1)  # (b, h, w, c) flat
    scratch_flat = scratch.permute(0, 2, 3, 1).contiguous().view(-1)  # will be
    # but scratch itself is dense (C,H,W) which is not NHWC. Simplest: alloc a
    # NHWC dense buffer.
    nhwc = torch.empty(
        (batch, height, width, channels),
        device=x.device,
        dtype=torch.bfloat16,
    )
    nhwc_flat = nhwc.view(-1)

    total = batch * channels * height * width
    # Pick a block that divides total AND is a multiple of C (so gather
    # `lane % C` is well defined on integer arithmetic — actually any
    # BLOCK works because we compute c_idx per-lane).
    for cand in (1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1):
        if total % cand == 0:
            block = cand
            break

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(total, block), 1, 1),
        _bn_silu_kernel,
        (x_perm, mean, var, weight, bias, nhwc_flat, channels, block, EPS),
    )

    # Copy nhwc (b,h,w,c) into bn_view (b, c, h, w) channels-last strides.
    # bn_view has shape (b, c, h, w) with strides (2C*H*W, 1, 2C*W, 2C) —
    # accepting a channels-last (b, h, w, c) source requires a copy_/permute.
    bn_view.copy_(nhwc.permute(0, 3, 1, 2))

    return out


@oracle_impl(hardware="B200", point="1ff5655c", BLOCK=256)
@oracle_impl(hardware="B200", point="b0704cab", BLOCK=256)
@oracle_impl(hardware="B200", point="91c4b41a", BLOCK=256)
def oracle_forward(inputs, *, BLOCK: int):
    return _launch(inputs, BLOCK=BLOCK)
