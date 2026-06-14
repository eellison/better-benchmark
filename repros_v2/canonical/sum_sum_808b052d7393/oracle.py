"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete GhostNet bf16 adaptive-average-pool backward, rounded ReLU gate, two BatchNorm-backward channel reductions, f32 side vector, and layout-preserving bf16 input-gradient output directly from the pooled `[512, 960, 1, 1]` source and channels-last activation, whereas Inductor materializes the zero-fill `as_strided_scatter -> as_strided -> expand -> div` producer and then schedules the gate, sibling reductions, and dependent epilogue as generic tensor work; Inductor cannot do this today because scheduler/codegen does not model zero-fill view/as_strided scatter followed by broadcasted average-pool backward as a structured scatter-reduce source that can feed both reductions and the required strided side-output stores while preserving bf16 rounding boundaries; the fix is SCATTER_REDUCE: add a structured average-pool-backward scatter-reduce lowering that maps pooled-gradient source elements directly into the ReLU-gated BN reduction template and emits the dependent output tensor in the target layout."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _div_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _partials_kernel(
    pooled_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    scalar_ptr,
    partial_sum_ptr,
    partial_centered_ptr,
    pooled_stride_n: tl.constexpr,
    pooled_stride_c: tl.constexpr,
    act_stride_n: tl.constexpr,
    act_stride_c: tl.constexpr,
    act_stride_h: tl.constexpr,
    act_stride_w: tl.constexpr,
    mean_stride_c: tl.constexpr,
    invstd_stride_c: tl.constexpr,
    weight_stride_c: tl.constexpr,
    bias_stride_c: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    W: tl.constexpr,
    REDUCE_SIZE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    tile = tl.program_id(0)
    c_block = tl.program_id(1)
    rows = tile * BLOCK_R + tl.arange(0, BLOCK_R)[:, None]
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    col_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    active = (rows < REDUCE_SIZE) & (cols < C)
    col_mask = col_offsets < C

    n = rows // HW
    spatial = rows - n * HW
    h = spatial // W
    w = spatial - h * W

    activation_offsets = (
        n * act_stride_n
        + cols * act_stride_c
        + h * act_stride_h
        + w * act_stride_w
    )
    pooled_offsets = n * pooled_stride_n + cols * pooled_stride_c

    mean = tl.load(mean_ptr + col_offsets * mean_stride_c, mask=col_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + col_offsets * invstd_stride_c, mask=col_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + col_offsets * weight_stride_c, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets * bias_stride_c, mask=col_mask, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr)

    activation = tl.load(activation_ptr + activation_offsets, mask=active, other=0.0).to(tl.float32)
    centered = _sub_rn_f32(activation, mean[None, :])

    affine = _mul_rn_f32(centered, invstd[None, :])
    affine = _mul_rn_f32(affine, weight[None, :])
    affine = _add_rn_f32(affine, bias[None, :])
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    affine_gate = affine_bf16.to(tl.float32)
    take_pool = active & ((affine_gate > 0.0) | (affine_gate != affine_gate))

    pooled = tl.load(pooled_ptr + pooled_offsets, mask=active, other=0.0).to(tl.float32)
    pool_bf16 = _div_rn_f32(pooled, 49.0).to(tl.bfloat16, fp_downcast_rounding="rtne")
    selected = tl.where(take_pool, pool_bf16, scalar).to(tl.float32)
    selected = tl.where(active, selected, 0.0)
    centered = tl.where(active, centered, 0.0)

    partial_sum = tl.sum(selected, axis=0)
    partial_centered = tl.sum(_mul_rn_f32(selected, centered), axis=0)
    store_offsets = tile * C + col_offsets
    tl.store(partial_sum_ptr + store_offsets, partial_sum, mask=col_mask)
    tl.store(partial_centered_ptr + store_offsets, partial_centered, mask=col_mask)


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_centered_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    vector_out_ptr,
    coeff_mean_ptr,
    coeff_var_ptr,
    coeff_weight_ptr,
    invstd_stride_c: tl.constexpr,
    weight_stride_c: tl.constexpr,
    C: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)[:, None]
    active = (tiles < NUM_TILES) & (cols[None, :] < C)
    offsets = tiles * C + cols[None, :]

    partial_sum = tl.load(partial_sum_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    partial_centered = tl.load(partial_centered_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    sum_where = tl.sum(partial_sum, axis=0)
    sum_centered = tl.sum(partial_centered, axis=0)

    col_mask = cols < C
    invstd = tl.load(invstd_ptr + cols * invstd_stride_c, mask=col_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols * weight_stride_c, mask=col_mask, other=0.0).to(tl.float32)

    scaled_sum = _mul_rn_f32(sum_where, 3.985969387755102e-05)
    scaled_centered = _mul_rn_f32(sum_centered, 3.985969387755102e-05)
    invstd_sq = _mul_rn_f32(invstd, invstd)
    coeff_var = _mul_rn_f32(scaled_centered, invstd_sq)
    coeff_weight = _mul_rn_f32(invstd, weight)
    vector_out = _mul_rn_f32(sum_centered, invstd)

    tl.store(sum_out_ptr + cols, sum_where, mask=col_mask)
    tl.store(vector_out_ptr + cols, vector_out, mask=col_mask)
    tl.store(coeff_mean_ptr + cols, scaled_sum, mask=col_mask)
    tl.store(coeff_var_ptr + cols, coeff_var, mask=col_mask)
    tl.store(coeff_weight_ptr + cols, coeff_weight, mask=col_mask)


@triton.jit
def _epilogue_kernel(
    pooled_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    scalar_ptr,
    coeff_mean_ptr,
    coeff_var_ptr,
    coeff_weight_ptr,
    out_ptr,
    pooled_stride_n: tl.constexpr,
    pooled_stride_c: tl.constexpr,
    act_stride_n: tl.constexpr,
    act_stride_c: tl.constexpr,
    act_stride_h: tl.constexpr,
    act_stride_w: tl.constexpr,
    mean_stride_c: tl.constexpr,
    invstd_stride_c: tl.constexpr,
    weight_stride_c: tl.constexpr,
    bias_stride_c: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    W: tl.constexpr,
    NUMEL: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    active = linear < NUMEL

    c = linear % C
    nhw = linear // C
    spatial = nhw % HW
    n = nhw // HW
    h = spatial // W
    w = spatial - h * W

    activation_offsets = (
        n * act_stride_n
        + c * act_stride_c
        + h * act_stride_h
        + w * act_stride_w
    )
    pooled_offsets = n * pooled_stride_n + c * pooled_stride_c
    out_offsets = (
        n * out_stride_n
        + c * out_stride_c
        + h * out_stride_h
        + w * out_stride_w
    )

    mean = tl.load(mean_ptr + c * mean_stride_c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c * invstd_stride_c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c * weight_stride_c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c * bias_stride_c, mask=active, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr)

    activation = tl.load(activation_ptr + activation_offsets, mask=active, other=0.0).to(tl.float32)
    centered = _sub_rn_f32(activation, mean)

    affine = _mul_rn_f32(centered, invstd)
    affine = _mul_rn_f32(affine, weight)
    affine = _add_rn_f32(affine, bias)
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    affine_gate = affine_bf16.to(tl.float32)
    take_pool = active & ((affine_gate > 0.0) | (affine_gate != affine_gate))

    pooled = tl.load(pooled_ptr + pooled_offsets, mask=active, other=0.0).to(tl.float32)
    pool_bf16 = _div_rn_f32(pooled, 49.0).to(tl.bfloat16, fp_downcast_rounding="rtne")
    selected = tl.where(take_pool, pool_bf16, scalar).to(tl.float32)

    coeff_mean = tl.load(coeff_mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    coeff_var = tl.load(coeff_var_ptr + c, mask=active, other=0.0).to(tl.float32)
    coeff_weight = tl.load(coeff_weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    variance_term = _mul_rn_f32(centered, coeff_var)
    without_var = _sub_rn_f32(selected, variance_term)
    without_mean = _sub_rn_f32(without_var, coeff_mean)
    result = _mul_rn_f32(without_mean, coeff_weight)
    tl.store(out_ptr + out_offsets, result.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=active)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="69efee57",
    BLOCK_R=128,
    BLOCK_C=16,
    BLOCK_E=256,
    reduce_warps=4,
    epilogue_warps=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    BLOCK_E: int,
    reduce_warps: int,
    epilogue_warps: int,
):
    (
        pooled,
        activation,
        mean,
        invstd,
        weight,
        bias,
        scalar,
        _shape0,
        _shape1,
        _shape2,
        _shape3,
        _shape4,
        _shape5,
    ) = inputs

    n, channels, height, width = (int(dim) for dim in activation.shape)
    hw = height * width
    reduce_size = n * hw
    numel = n * channels * hw
    num_tiles = triton.cdiv(reduce_size, BLOCK_R)
    block_tiles = _next_power_of_2(num_tiles)

    partial_sum = torch.empty((num_tiles, channels), device=activation.device, dtype=torch.float32)
    partial_centered = torch.empty((num_tiles, channels), device=activation.device, dtype=torch.float32)
    sum_out = torch.empty((channels,), device=activation.device, dtype=torch.float32)
    vector_out = torch.empty((channels,), device=activation.device, dtype=torch.float32)
    coeff_mean = torch.empty((channels,), device=activation.device, dtype=torch.float32)
    coeff_var = torch.empty((channels,), device=activation.device, dtype=torch.float32)
    coeff_weight = torch.empty((channels,), device=activation.device, dtype=torch.float32)
    out = torch.empty_strided(
        tuple(activation.shape),
        tuple(int(s) for s in activation.stride()),
        device=activation.device,
        dtype=torch.bfloat16,
    )

    common = dict(
        pooled_stride_n=int(pooled.stride(0)),
        pooled_stride_c=int(pooled.stride(1)),
        act_stride_n=int(activation.stride(0)),
        act_stride_c=int(activation.stride(1)),
        act_stride_h=int(activation.stride(2)),
        act_stride_w=int(activation.stride(3)),
        mean_stride_c=int(mean.stride(1)),
        invstd_stride_c=int(invstd.stride(1)),
        weight_stride_c=int(weight.stride(0)),
        bias_stride_c=int(bias.stride(0)),
        C=channels,
        HW=hw,
        W=width,
    )
    _partials_kernel[(num_tiles, triton.cdiv(channels, BLOCK_C))](
        pooled,
        activation,
        mean,
        invstd,
        weight,
        bias,
        scalar,
        partial_sum,
        partial_centered,
        **common,
        REDUCE_SIZE=reduce_size,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=reduce_warps,
        num_stages=3,
    )
    _finalize_kernel[(triton.cdiv(channels, BLOCK_C),)](
        partial_sum,
        partial_centered,
        invstd,
        weight,
        sum_out,
        vector_out,
        coeff_mean,
        coeff_var,
        coeff_weight,
        invstd_stride_c=int(invstd.stride(1)),
        weight_stride_c=int(weight.stride(0)),
        C=channels,
        NUM_TILES=num_tiles,
        BLOCK_TILES=block_tiles,
        BLOCK_C=BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _epilogue_kernel[(triton.cdiv(numel, BLOCK_E),)](
        pooled,
        activation,
        mean,
        invstd,
        weight,
        bias,
        scalar,
        coeff_mean,
        coeff_var,
        coeff_weight,
        out,
        **common,
        out_stride_n=int(out.stride(0)),
        out_stride_c=int(out.stride(1)),
        out_stride_h=int(out.stride(2)),
        out_stride_w=int(out.stride(3)),
        NUMEL=numel,
        BLOCK_E=BLOCK_E,
        num_warps=epilogue_warps,
        num_stages=3,
    )
    return sum_out, vector_out, out
