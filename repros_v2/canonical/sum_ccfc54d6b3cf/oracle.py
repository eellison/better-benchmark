"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete bf16 max-pool-backward scatter_add, mask overwrite, returned masked tensor, and returned per-channel sum by reverse-gathering the low-memory max-pool offsets directly into the dense output, whereas Inductor materializes the f32 scatter_add buffer, casts it to bf16, applies the where, and then launches a generic channel reduction. Inductor cannot do this today because scheduler/codegen treats `_low_memory_max_pool_offsets_to_indices` plus `scatter_add` as an opaque producer instead of a structured max-pool scatter-reduce that can feed both a dense side output and a channel reduction while preserving bf16 cast boundaries; the fix is SCATTER_REDUCE: add a max-pool-backward lowering that derives each destination from the small set of candidate pooled sources and fuses the mask/reduction consumers."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _add_candidate(
    src_ptr,
    offset_ptr,
    nc_base,
    ph,
    pw,
    oh,
    ow,
    active,
    PH: tl.constexpr,
    PW: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
):
    local_h = oh - ph * STRIDE
    local_w = ow - pw * STRIDE
    candidate = (
        active
        & (ph >= 0)
        & (ph < PH)
        & (pw >= 0)
        & (pw < PW)
        & (local_h >= 0)
        & (local_h < K)
        & (local_w >= 0)
        & (local_w < K)
    )
    src_index = nc_base + ph * PW + pw
    expected = (local_h * K + local_w).to(tl.int32)
    actual = tl.load(offset_ptr + src_index, mask=candidate, other=-1).to(tl.int32)
    return tl.load(src_ptr + src_index, mask=candidate & (actual == expected), other=0.0).to(tl.float32)


@triton.jit
def _materialize_where_kernel(
    src_ptr,
    offset_ptr,
    mask_ptr,
    fill_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL

    out_hw: tl.constexpr = OH * OW
    src_hw: tl.constexpr = PH * PW
    nc = offsets // out_hw
    spatial = offsets - nc * out_hw
    oh = spatial // OW
    ow = spatial - oh * OW
    nc_base = nc * src_hw

    ph0 = oh // STRIDE
    pw0 = ow // STRIDE
    acc = _add_candidate(src_ptr, offset_ptr, nc_base, ph0, pw0, oh, ow, active, PH, PW, K, STRIDE)
    if K == 3:
        acc += _add_candidate(src_ptr, offset_ptr, nc_base, ph0 - 1, pw0, oh, ow, active, PH, PW, K, STRIDE)
        acc += _add_candidate(src_ptr, offset_ptr, nc_base, ph0, pw0 - 1, oh, ow, active, PH, PW, K, STRIDE)
        acc += _add_candidate(src_ptr, offset_ptr, nc_base, ph0 - 1, pw0 - 1, oh, ow, active, PH, PW, K, STRIDE)

    scatter_value = acc.to(tl.bfloat16, fp_downcast_rounding="rtne")
    fill_value = tl.load(fill_ptr).to(tl.bfloat16)
    take_fill = tl.load(mask_ptr + offsets, mask=active, other=0) != 0
    out_value = tl.where(take_fill, fill_value, scatter_value)
    tl.store(out_ptr + offsets, out_value, mask=active)


@triton.jit
def _channel_sum_kernel(
    out_ptr,
    sum_ptr,
    C: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    channel = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    active = rows < R
    spatial_hw: tl.constexpr = OH * OW
    n = rows // spatial_hw
    spatial = rows - n * spatial_hw
    offsets = n * (C * spatial_hw) + channel * spatial_hw + spatial
    values = tl.load(out_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    total = tl.sum(tl.where(active, values, 0.0), axis=0)
    tl.store(sum_ptr + channel, total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32))


def _launch(
    inputs,
    *,
    N: int,
    C: int,
    PH: int,
    PW: int,
    OH: int,
    OW: int,
    K: int,
    STRIDE: int,
    BLOCK: int,
    BLOCK_R: int,
    materialize_warps: int,
    sum_warps: int,
):
    src, offsets, mask, fill = inputs[:4]
    out = torch.empty_strided(
        (N, C, OH, OW),
        (C * OH * OW, OH * OW, OW, 1),
        device=src.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided((C,), (1,), device=src.device, dtype=torch.float32)
    total = N * C * OH * OW
    _materialize_where_kernel[(triton.cdiv(total, BLOCK),)](
        src,
        offsets,
        mask,
        fill,
        out,
        TOTAL=total,
        C=C,
        PH=PH,
        PW=PW,
        OH=OH,
        OW=OW,
        K=K,
        STRIDE=STRIDE,
        BLOCK=BLOCK,
        num_warps=materialize_warps,
        num_stages=3,
    )
    _channel_sum_kernel[(C,)](
        out,
        sum_out,
        C=C,
        OH=OH,
        OW=OW,
        R=N * OH * OW,
        BLOCK_R=BLOCK_R,
        num_warps=sum_warps,
        num_stages=3,
    )
    return out, sum_out


# e1c3cc78: alexnet train, bf16 [128,256,6,6] max-pool scatter to [128,256,13,13].
@oracle_impl(
    hardware="B200",
    point="e1c3cc78",
    N=128,
    C=256,
    PH=6,
    PW=6,
    OH=13,
    OW=13,
    K=3,
    STRIDE=2,
    BLOCK=256,
    BLOCK_R=32768,
    materialize_warps=4,
    sum_warps=8,
)
# 2a69936f: vgg16 train, bf16 [64,512,7,7] max-pool scatter to [64,512,14,14].
@oracle_impl(
    hardware="B200",
    point="2a69936f",
    N=64,
    C=512,
    PH=7,
    PW=7,
    OH=14,
    OW=14,
    K=2,
    STRIDE=2,
    BLOCK=256,
    BLOCK_R=16384,
    materialize_warps=4,
    sum_warps=8,
)
def oracle_forward(inputs, **kwargs):
    return _launch(inputs, **kwargs)
