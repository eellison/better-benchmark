"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 batch-norm backward tail, including the bf16 `where` producer, two sibling fp32 channel reductions, dependent normalization epilogue, explicit bf16 rounding before the residual add, the full returned `[128,112,H,W]` add tensor, and the returned `[128,16,H,W]` slice alias, whereas Inductor lowers the captured where/cast/sum/sum/pointwise/add/slice graph through generic reduction and pointwise scheduler fragments; Inductor cannot do this today because its scheduler does not keep a shared masked producer, sibling reductions, dependent epilogue, full side-output materialization, and alias-only slice return in one fused plan while preserving bf16 cast boundaries; the fix is SCHEDULER_FUSION: teach the reduction scheduler to fuse compatible sibling channel reductions with dependent pointwise epilogues and visible aliasing side outputs."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _dual_reduce_partial_kernel(
    mask_input_ptr,
    fill_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    partial_sum1_ptr,
    partial_sum2_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    TOTAL_SPATIAL: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    active = k < TOTAL_SPATIAL

    n = k // HW
    hw = k - n * HW
    offsets = n * (C * HW) + c * HW + hw

    mask_input = tl.load(mask_input_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)
    source = tl.load(source_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    centered_source = tl.load(centered_source_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)

    where_value = tl.where(mask_input <= 0.0, fill, source)
    centered = _f32_sub(centered_source, mean)
    product = _f32_mul(where_value, centered)

    partial_offset = c * NUM_TILES + tile
    tl.store(partial_sum1_ptr + partial_offset, tl.sum(where_value, axis=0))
    tl.store(partial_sum2_ptr + partial_offset, tl.sum(product, axis=0))


@triton.jit
def _finalize_reduce_kernel(
    partial_sum1_ptr,
    partial_sum2_ptr,
    scale_ptr,
    sum1_ptr,
    sum2_ptr,
    mul8_ptr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    active = tiles < NUM_TILES
    offsets = c * NUM_TILES + tiles

    sum1_values = tl.load(partial_sum1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    sum2_values = tl.load(partial_sum2_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    sum1 = tl.sum(sum1_values, axis=0)
    sum2 = tl.sum(sum2_values, axis=0)
    scale = tl.load(scale_ptr + c).to(tl.float32)

    tl.store(sum1_ptr + c, sum1)
    tl.store(sum2_ptr + c, sum2)
    tl.store(mul8_ptr + c, _f32_mul(sum2, scale))


@triton.jit
def _bn_tail_epilogue_kernel(
    residual_ptr,
    mask_input_ptr,
    fill_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    scale_ptr,
    affine_weight_ptr,
    sum1_ptr,
    sum2_ptr,
    add_out_ptr,
    C: tl.constexpr,
    RESIDUAL_C: tl.constexpr,
    HW: tl.constexpr,
    TOTAL: tl.constexpr,
    INV_COUNT: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    out_offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = out_offsets < TOTAL

    hw = out_offsets % HW
    c = (out_offsets // HW) % C
    n = out_offsets // (C * HW)
    input_offsets = n * (C * HW) + c * HW + hw
    residual_offsets = n * (RESIDUAL_C * HW) + (c + 16) * HW + hw

    mask_input = tl.load(mask_input_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)
    source = tl.load(source_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    centered_source = tl.load(centered_source_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=active, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + residual_offsets, mask=active, other=0.0).to(tl.float32)

    where_value = tl.where(mask_input <= 0.0, fill, source)
    centered = _f32_sub(centered_source, mean)
    mean_term = _f32_mul(sum1, INV_COUNT)
    sum2_scaled = _f32_mul(sum2, INV_COUNT)
    scale_square = _f32_mul(scale, scale)
    variance_term = _f32_mul(sum2_scaled, scale_square)
    affine = _f32_mul(scale, affine_weight)

    centered_scaled = _f32_mul(centered, variance_term)
    sub1 = _f32_sub(where_value, centered_scaled)
    sub2 = _f32_sub(sub1, mean_term)
    normalized = _f32_mul(sub2, affine)
    normalized_bf16 = normalized.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    add_value = _f32_add(residual, normalized_bf16)
    tl.store(
        add_out_ptr + out_offsets,
        add_value.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=active,
    )


def _contiguous_nchw_stride(batch: int, channels: int, height: int, width: int):
    return (channels * height * width, height * width, width, 1)


# 0b002951: (T([128,128,32,32], bf16), T([128,112,32,32], bf16), ...)
@oracle_impl(hardware="B200", point="0b002951", BLOCK_K=1024, BLOCK_ELEMS=256, num_warps_reduce=4, num_warps_finalize=8, num_warps_epilogue=4)
# 21952d3a: (T([128,128,16,16], bf16), T([128,112,16,16], bf16), ...)
@oracle_impl(hardware="B200", point="21952d3a", BLOCK_K=1024, BLOCK_ELEMS=256, num_warps_reduce=4, num_warps_finalize=4, num_warps_epilogue=4)
# 75c39973: (T([128,128,8,8], bf16), T([128,112,8,8], bf16), ...)
@oracle_impl(hardware="B200", point="75c39973", BLOCK_K=1024, BLOCK_ELEMS=256, num_warps_reduce=4, num_warps_finalize=4, num_warps_epilogue=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    BLOCK_ELEMS: int,
    num_warps_reduce: int,
    num_warps_finalize: int,
    num_warps_epilogue: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    batch = int(arg1_1.shape[0])
    channels = int(arg1_1.shape[1])
    height = int(arg1_1.shape[2])
    width = int(arg1_1.shape[3])
    residual_channels = int(arg0_1.shape[1])
    hw = height * width
    total_spatial = batch * hw
    total = batch * channels * hw
    num_tiles = triton.cdiv(total_spatial, BLOCK_K)
    block_tiles = triton.next_power_of_2(num_tiles)

    partial_sum1 = torch.empty((channels, num_tiles), device=arg1_1.device, dtype=torch.float32)
    partial_sum2 = torch.empty((channels, num_tiles), device=arg1_1.device, dtype=torch.float32)
    sum1 = torch.empty((channels,), device=arg1_1.device, dtype=torch.float32)
    sum2 = torch.empty((channels,), device=arg1_1.device, dtype=torch.float32)
    mul8 = torch.empty((channels,), device=arg1_1.device, dtype=torch.float32)
    add_out = torch.empty_strided(
        (batch, channels, height, width),
        _contiguous_nchw_stride(batch, channels, height, width),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    _dual_reduce_partial_kernel[(channels, num_tiles)](
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        partial_sum1,
        partial_sum2,
        C=channels,
        HW=hw,
        TOTAL_SPATIAL=total_spatial,
        NUM_TILES=num_tiles,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps_reduce,
    )

    _finalize_reduce_kernel[(channels,)](
        partial_sum1,
        partial_sum2,
        arg6_1,
        sum1,
        sum2,
        mul8,
        NUM_TILES=num_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=num_warps_finalize,
    )

    _bn_tail_epilogue_kernel[(triton.cdiv(total, BLOCK_ELEMS),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        sum1,
        sum2,
        add_out,
        C=channels,
        RESIDUAL_C=residual_channels,
        HW=hw,
        TOTAL=total,
        INV_COUNT=7.62939453125e-06,
        BLOCK_ELEMS=BLOCK_ELEMS,
        num_warps=num_warps_epilogue,
    )

    slice2 = torch.as_strided(
        add_out,
        (batch, 16, height, width),
        add_out.stride(),
        0,
    )
    return sum1, mul8, add_out, slice2
