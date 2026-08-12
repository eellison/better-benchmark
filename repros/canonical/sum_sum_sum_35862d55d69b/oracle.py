"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXtV2 layer-norm-backward tail, including the strided f32+bf16 residual add, the NCHW-to-NHWC logical layout, the per-pixel weighted channel reductions, the two sibling f32 channel reductions, the returned bf16 tensor with its captured nonstandard stride, and the final bf16-rounded channel sum from that tensor in one fused producer plus finalizer, whereas Inductor lowers the add, permutes, row reductions, dependent pointwise algebra, layout permute, and channel reductions as separate generic scheduled regions; Inductor cannot do this today because its scheduler does not form one multi-output reduction plan across reductions with different axes that share strided channels-last producers and a required bf16 materialization boundary; the fix is SCHEDULER_FUSION: teach reduction scheduling to compute the channel row summaries once, reuse them for all channel partials, and sink the required strided bf16 output store plus bf16-sum partials into the same producer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


def _ceil_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@triton.jit
def _producer_kernel(
    x_bf16_ptr,
    residual_ptr,
    weight_ptr,
    grad_bf16_ptr,
    mean_ptr,
    scale_ptr,
    y_ptr,
    partials_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    ROWS: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    XS3: tl.constexpr,
    RS0: tl.constexpr,
    RS1: tl.constexpr,
    RS2: tl.constexpr,
    RS3: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    GS3: tl.constexpr,
    MS0: tl.constexpr,
    MS1: tl.constexpr,
    MS2: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    YS0: tl.constexpr,
    YS1: tl.constexpr,
    YS2: tl.constexpr,
    YS3: tl.constexpr,
    NUM_ROW_TILES: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    tile = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)[:, None]
    r = tile * BLOCK_R + tl.arange(0, BLOCK_R)[None, :]
    active = (c < C) & (r < ROWS)

    hw = r % (H * W)
    n = r // (H * W)
    h = hw // W
    w = hw - h * W

    x_offsets = n * XS0 + c * XS1 + h * XS2 + w * XS3
    residual_offsets = n * RS0 + c * RS1 + h * RS2 + w * RS3
    grad_offsets = n * GS0 + c * GS1 + h * GS2 + w * GS3
    mean_offsets = n * MS0 + h * MS1 + w * MS2
    scale_offsets = n * SS0 + h * SS1 + w * SS2
    y_offsets = n * YS0 + c * YS1 + h * YS2 + w * YS3

    x_bf16 = tl.load(x_bf16_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + residual_offsets, mask=active, other=0.0).to(tl.float32)
    x = residual + x_bf16
    grad_src = tl.load(grad_bf16_ptr + grad_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + mean_offsets, mask=r < ROWS, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + scale_offsets, mask=r < ROWS, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c < C, other=0.0).to(tl.float32)

    normalized = (grad_src - mean) * scale
    weighted = x * weight
    row_weighted_sum = tl.sum(tl.where(active, weighted, 0.0), axis=0)
    row_weighted_norm_sum = tl.sum(tl.where(active, weighted * normalized, 0.0), axis=0)

    centered = weighted * C - row_weighted_sum[None, :]
    centered = centered - normalized * row_weighted_norm_sum[None, :]
    value = (scale / C) * centered
    value_bf16 = value.to(tl.bfloat16)
    tl.store(y_ptr + y_offsets, value_bf16, mask=active)

    c_vec = tl.arange(0, BLOCK_C)
    c_mask = c_vec < C
    partial_base = tile * C + c_vec
    partial_stride = NUM_ROW_TILES * C

    tl.store(
        partials_ptr + partial_base,
        tl.sum(tl.where(active, x * normalized, 0.0), axis=1),
        mask=c_mask,
    )
    tl.store(
        partials_ptr + partial_stride + partial_base,
        tl.sum(tl.where(active, x, 0.0), axis=1),
        mask=c_mask,
    )
    tl.store(
        partials_ptr + 2 * partial_stride + partial_base,
        tl.sum(tl.where(active, value_bf16.to(tl.float32), 0.0), axis=1),
        mask=c_mask,
    )


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    out_sum_xnorm_ptr,
    out_sum_x_ptr,
    out_sum_y_ptr,
    C: tl.constexpr,
    NUM_ROW_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    tiles = tl.arange(0, BLOCK_TILES)[:, None]
    active = (c < C) & (tiles < NUM_ROW_TILES)
    offsets = tiles * C + c
    partial_stride = NUM_ROW_TILES * C

    sum_xnorm = tl.sum(
        tl.load(partials_ptr + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )
    sum_x = tl.sum(
        tl.load(partials_ptr + partial_stride + offsets, mask=active, other=0.0).to(
            tl.float32
        ),
        axis=0,
    )
    sum_y = tl.sum(
        tl.load(partials_ptr + 2 * partial_stride + offsets, mask=active, other=0.0).to(
            tl.float32
        ),
        axis=0,
    )

    c_vec = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tl.store(out_sum_xnorm_ptr + c_vec, sum_xnorm, mask=c_vec < C)
    tl.store(out_sum_x_ptr + c_vec, sum_x, mask=c_vec < C)
    tl.store(out_sum_y_ptr + c_vec, sum_y.to(tl.bfloat16).to(tl.float32), mask=c_vec < C)


@oracle_impl(hardware="B200", point="5fae49ec", BLOCK_R=128, BLOCK_C=128, FINAL_BLOCK_C=8, num_warps=8)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    x_bf16, residual, weight, grad_bf16, mean, scale = inputs
    n = int(x_bf16.shape[0])
    c = int(x_bf16.shape[1])
    h = int(x_bf16.shape[2])
    w = int(x_bf16.shape[3])
    rows = n * h * w
    num_row_tiles = triton.cdiv(rows, BLOCK_R)
    device = x_bf16.device

    out0 = torch.empty((c,), device=device, dtype=torch.float32)
    out1 = torch.empty((c,), device=device, dtype=torch.float32)
    y = torch.empty_strided(
        (n, c, h, w),
        (
            int(residual.stride(0)),
            int(residual.stride(1)),
            int(residual.stride(2)),
            int(residual.stride(3)),
        ),
        device=device,
        dtype=torch.bfloat16,
    )
    out3 = torch.empty((c,), device=device, dtype=torch.float32)
    partials = torch.empty((3, num_row_tiles, c), device=device, dtype=torch.float32)

    _producer_kernel[(num_row_tiles,)](
        x_bf16,
        residual,
        weight,
        grad_bf16,
        mean,
        scale,
        y,
        partials,
        C=c,
        H=h,
        W=w,
        ROWS=rows,
        XS0=int(x_bf16.stride(0)),
        XS1=int(x_bf16.stride(1)),
        XS2=int(x_bf16.stride(2)),
        XS3=int(x_bf16.stride(3)),
        RS0=int(residual.stride(0)),
        RS1=int(residual.stride(1)),
        RS2=int(residual.stride(2)),
        RS3=int(residual.stride(3)),
        GS0=int(grad_bf16.stride(0)),
        GS1=int(grad_bf16.stride(1)),
        GS2=int(grad_bf16.stride(2)),
        GS3=int(grad_bf16.stride(3)),
        MS0=int(mean.stride(0)),
        MS1=int(mean.stride(1)),
        MS2=int(mean.stride(2)),
        SS0=int(scale.stride(0)),
        SS1=int(scale.stride(1)),
        SS2=int(scale.stride(2)),
        YS0=int(y.stride(0)),
        YS1=int(y.stride(1)),
        YS2=int(y.stride(2)),
        YS3=int(y.stride(3)),
        NUM_ROW_TILES=num_row_tiles,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=1,
    )
    _finalize_partials_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
        partials,
        out0,
        out1,
        out3,
        C=c,
        NUM_ROW_TILES=num_row_tiles,
        BLOCK_TILES=_ceil_pow2(num_row_tiles),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
        num_stages=1,
    )
    return out0, out1, y, out3
