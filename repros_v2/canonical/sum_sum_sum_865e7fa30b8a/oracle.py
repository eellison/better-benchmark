"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXtV2 GRN backward-style bf16 scope, including the NCHW-to-NHWC logical layout, the literal-80 GRN scale/divide captured in the graph, the per-pixel weighted channel reductions, the two sibling channel reductions, the bf16-rounded returned tensor with its captured nonstandard stride, and the final bf16 sum from that returned tensor in one fused partial-reduction producer plus a co-finalizer, whereas Inductor lowers the permutes, row reductions, pointwise GRN-gradient algebra, layout permute, and channel reductions as generic scheduled regions; Inductor cannot do this today because its scheduler does not form one multi-output reduction plan across reductions with different axes that share strided channels-last producers and dependent row summaries; the fix is SCHEDULER_FUSION: teach reduction scheduling to compute the channel row summaries once, reuse them for all channel partials, and sink required bf16 materialization into the same producer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


def _ceil_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@triton.jit
def _f32_add(a, b):
    return a + b


@triton.jit
def _f32_sub(a, b):
    return a - b


@triton.jit
def _f32_mul(a, b):
    return a * b


@triton.jit
def _f32_div(a, b):
    return a / b


@triton.jit
def _grn_producer_kernel(
    x_ptr,
    weight_ptr,
    grad_ptr,
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
    grad_offsets = n * GS0 + c * GS1 + h * GS2 + w * GS3
    row_offsets_mean = n * MS0 + h * MS1 + w * MS2
    row_offsets_scale = n * SS0 + h * SS1 + w * SS2

    x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
    grad_in = tl.load(grad_ptr + grad_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row_offsets_mean, mask=r < ROWS, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + row_offsets_scale, mask=r < ROWS, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c < C, other=0.0).to(tl.float32)

    centered = _f32_sub(grad_in, mean)
    grad_scaled = _f32_mul(centered, scale)
    weighted = _f32_mul(x, weight)
    weighted_grad = _f32_mul(weighted, grad_scaled)

    row_weighted_sum = tl.sum(tl.where(active, weighted, 0.0), axis=0)
    row_weighted_grad_sum = tl.sum(tl.where(active, weighted_grad, 0.0), axis=0)

    lhs = _f32_sub(_f32_mul(weighted, 80.0), row_weighted_sum[None, :])
    rhs = _f32_mul(grad_scaled, row_weighted_grad_sum[None, :])
    value = _f32_mul(_f32_div(scale, 80.0), _f32_sub(lhs, rhs))
    value_bf16 = value.to(tl.bfloat16)

    y_offsets = n * YS0 + c * YS1 + h * YS2 + w * YS3
    tl.store(y_ptr + y_offsets, value_bf16, mask=active)

    c_vec = tl.arange(0, BLOCK_C)
    c_mask = c_vec < C
    partial_base = tile * C + c_vec
    partial_stride = NUM_ROW_TILES * C

    tl.store(
        partials_ptr + partial_base,
        tl.sum(tl.where(active, _f32_mul(x, grad_scaled), 0.0), axis=1),
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
    out_sum_xgrad_ptr,
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

    sum_xgrad = tl.sum(
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
    tl.store(out_sum_xgrad_ptr + c_vec, sum_xgrad, mask=c_vec < C)
    tl.store(out_sum_x_ptr + c_vec, sum_x, mask=c_vec < C)
    tl.store(out_sum_y_ptr + c_vec, sum_y.to(tl.bfloat16).to(tl.float32), mask=c_vec < C)


def _launch(inputs, *, BLOCK_R, BLOCK_C, FINAL_BLOCK_C, num_warps):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    rows = n * h * w
    num_row_tiles = triton.cdiv(rows, BLOCK_R)

    out0 = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)
    out1 = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)
    y = torch.empty_strided(
        (n, c, h, w),
        (int(arg0_1.stride(0)), int(arg0_1.stride(1)), int(arg0_1.stride(3)), int(arg0_1.stride(2))),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out3 = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)
    partials = torch.empty((3, num_row_tiles, c), device=arg0_1.device, dtype=torch.float32)

    _grn_producer_kernel[(num_row_tiles,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        y,
        partials,
        C=c,
        H=h,
        W=w,
        ROWS=rows,
        XS0=int(arg0_1.stride(0)),
        XS1=int(arg0_1.stride(1)),
        XS2=int(arg0_1.stride(2)),
        XS3=int(arg0_1.stride(3)),
        GS0=int(arg2_1.stride(0)),
        GS1=int(arg2_1.stride(1)),
        GS2=int(arg2_1.stride(2)),
        GS3=int(arg2_1.stride(3)),
        MS0=int(arg3_1.stride(0)),
        MS1=int(arg3_1.stride(1)),
        MS2=int(arg3_1.stride(2)),
        SS0=int(arg4_1.stride(0)),
        SS1=int(arg4_1.stride(1)),
        SS2=int(arg4_1.stride(2)),
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


# ConvNeXtV2 nano GRN backward, C=80, H=W=56.
@oracle_impl(hardware="B200", point="5cee1fdd", BLOCK_R=128, BLOCK_C=128, FINAL_BLOCK_C=8, num_warps=8)
# ConvNeXtV2 nano GRN backward, C=160, H=W=28.
@oracle_impl(hardware="B200", point="9deba5e9", BLOCK_R=32, BLOCK_C=256, FINAL_BLOCK_C=2, num_warps=4)
# ConvNeXtV2 nano GRN backward, C=320, H=W=14.
@oracle_impl(hardware="B200", point="2cbe1bb4", BLOCK_R=16, BLOCK_C=512, FINAL_BLOCK_C=16, num_warps=4)
# ConvNeXtV2 nano GRN backward, C=640, H=W=7.
@oracle_impl(hardware="B200", point="d5a3644b", BLOCK_R=16, BLOCK_C=1024, FINAL_BLOCK_C=16, num_warps=8)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    return _launch(
        inputs,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        FINAL_BLOCK_C=FINAL_BLOCK_C,
        num_warps=num_warps,
    )
