"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 masked avg-pool-to-BatchNorm-backward tail, including the bf16 divide-by-49 and `where` output, both fp32 per-channel sums over N/H/W, the f32 side vector, and the final bf16 full-tensor epilogue while preserving the captured cast boundaries, whereas Inductor lowers the decomposed view/expand/div/where/cast/sibling-reduction/dependent-epilogue graph as generic producer, reduction, and pointwise kernels; Inductor cannot do this today because its scheduler does not form one full-scope multi-output channel-reduction plan that shares the bf16 masked producer across sibling sums and sinks the finalized channel scalars into the returned tensor epilogue; the fix is SCHEDULER_FUSION: teach reduction scheduling to fuse compatible sibling channel reductions with their live producer output, side vector, and dependent bf16 epilogue stores."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


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
def _masked_dual_sum_kernel(
    pooled_grad_ptr,
    mask_ptr,
    activation_ptr,
    mean_ptr,
    where_out_ptr,
    partial_sum_ptr,
    partial_centered_sum_ptr,
    act_stride_n: tl.constexpr,
    act_stride_c: tl.constexpr,
    act_stride_h: tl.constexpr,
    act_stride_w: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    REDUCE_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    r = tile * BLOCK_R + tl.arange(0, BLOCK_R)
    active = r < REDUCE_SIZE

    n = r // HW
    hw = r - n * HW
    h = hw // W
    w = hw - h * W

    contiguous_offsets = n * (C * HW) + c * HW + hw
    pooled_offsets = n * C + c
    activation_offsets = (
        n * act_stride_n
        + c * act_stride_c
        + h * act_stride_h
        + w * act_stride_w
    )

    pooled = tl.load(pooled_grad_ptr + pooled_offsets, mask=active, other=0.0).to(tl.float32)
    divided_bf16 = (pooled / 49.0).to(tl.bfloat16)
    zero_bf16 = tl.full((BLOCK_R,), 0.0, tl.bfloat16)
    pred = tl.load(mask_ptr + contiguous_offsets, mask=active, other=1)
    where_bf16 = tl.where(pred, zero_bf16, divided_bf16)
    tl.store(where_out_ptr + contiguous_offsets, where_bf16, mask=active)

    where_f32 = where_bf16.to(tl.float32)
    activation = tl.load(
        activation_ptr + activation_offsets,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = _sub_rn_f32(activation, mean)
    product = _mul_rn_f32(where_f32, centered)

    partial_offset = c * NUM_TILES + tile
    tl.store(
        partial_sum_ptr + partial_offset,
        tl.sum(tl.where(active, where_f32, 0.0), axis=0),
    )
    tl.store(
        partial_centered_sum_ptr + partial_offset,
        tl.sum(tl.where(active, product, 0.0), axis=0),
    )


@triton.jit
def _finalize_channel_kernel(
    partial_sum_ptr,
    partial_centered_sum_ptr,
    invstd_ptr,
    weight_ptr,
    full_out_ptr,
    sum_out_ptr,
    vector_out_ptr,
    coeff_mean_ptr,
    coeff_var_ptr,
    coeff_weight_ptr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = tiles < NUM_TILES
    offsets = c * NUM_TILES + tiles

    sum_values = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    centered_values = tl.load(
        partial_centered_sum_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sum_where = tl.sum(sum_values, axis=0)
    sum_centered = tl.sum(centered_values, axis=0)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)

    scaled_sum = _mul_rn_f32(sum_where, 0.0012755102040816326)
    scaled_centered = _mul_rn_f32(sum_centered, 0.0012755102040816326)
    invstd_sq = _mul_rn_f32(invstd, invstd)
    coeff_var = _mul_rn_f32(scaled_centered, invstd_sq)
    coeff_weight = _mul_rn_f32(invstd, weight)
    vector_out = _mul_rn_f32(sum_centered, invstd)

    tl.store(sum_out_ptr + c, sum_where)
    tl.store(vector_out_ptr + c, vector_out)
    tl.store(coeff_mean_ptr + c, scaled_sum)
    tl.store(coeff_var_ptr + c, coeff_var)
    tl.store(coeff_weight_ptr + c, coeff_weight)
    tl.store(full_out_ptr, tl.full((), 0.0, tl.bfloat16), mask=c == 0)


@triton.jit
def _bn_backward_epilogue_kernel(
    where_ptr,
    activation_ptr,
    mean_ptr,
    coeff_mean_ptr,
    coeff_var_ptr,
    coeff_weight_ptr,
    out_ptr,
    act_stride_n: tl.constexpr,
    act_stride_c: tl.constexpr,
    act_stride_h: tl.constexpr,
    act_stride_w: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    NUMEL: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    active = offsets < NUMEL

    hw = offsets % HW
    c = (offsets // HW) % C
    n = offsets // (C * HW)
    h = hw // W
    w = hw - h * W
    activation_offsets = (
        n * act_stride_n
        + c * act_stride_c
        + h * act_stride_h
        + w * act_stride_w
    )

    where_value = tl.load(where_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    activation = tl.load(
        activation_ptr + activation_offsets,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    coeff_mean = tl.load(coeff_mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    coeff_var = tl.load(coeff_var_ptr + c, mask=active, other=0.0).to(tl.float32)
    coeff_weight = tl.load(coeff_weight_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = _sub_rn_f32(activation, mean)
    variance_term = _mul_rn_f32(centered, coeff_var)
    without_var = _sub_rn_f32(where_value, variance_term)
    without_mean = _sub_rn_f32(without_var, coeff_mean)
    result = _mul_rn_f32(without_mean, coeff_weight)
    tl.store(out_ptr + offsets, result.to(tl.bfloat16), mask=active)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# 78158bd3: resnet18, N=16, C=512, H=W=7.
@oracle_impl(hardware="B200", point="78158bd3", BLOCK_R=1024, BLOCK_E=256, reduce_warps=4, epilogue_warps=4)
# 0a877abf: resnet50, N=32, C=2048, H=W=7.
@oracle_impl(hardware="B200", point="0a877abf", BLOCK_R=2048, BLOCK_E=256, reduce_warps=8, epilogue_warps=4)
# 186ca521: resnext50_32x4d, N=8, C=2048, H=W=7.
@oracle_impl(hardware="B200", point="186ca521", BLOCK_R=512, BLOCK_E=256, reduce_warps=4, epilogue_warps=4)
# b0156f06: phlippe_resnet, N=128, C=64, H=W=8.
@oracle_impl(hardware="B200", point="b0156f06", BLOCK_R=1024, BLOCK_E=256, reduce_warps=4, epilogue_warps=4)
# 50796f54: resnet152, N=128, C=2048, H=W=7.
@oracle_impl(hardware="B200", point="50796f54", BLOCK_R=2048, BLOCK_E=256, reduce_warps=8, epilogue_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_E: int,
    reduce_warps: int,
    epilogue_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape0, _shape1 = inputs
    shape = tuple(int(dim) for dim in arg1_1.shape)
    n, channels, height, width = shape
    hw = height * width
    reduce_size = n * hw
    numel = n * channels * hw
    num_tiles = triton.cdiv(reduce_size, BLOCK_R)
    block_tiles = triton.next_power_of_2(num_tiles)

    full = torch.empty_strided((), (), device=arg0_1.device, dtype=torch.bfloat16)
    where = torch.empty_strided(
        shape,
        _contiguous_stride(shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided(
        (channels,),
        (1,),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    vector_out = torch.empty_strided(
        (channels,),
        (1,),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        shape,
        _contiguous_stride(shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty_strided(
        (channels, num_tiles),
        (num_tiles, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    partial_centered_sum = torch.empty_strided(
        (channels, num_tiles),
        (num_tiles, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    coeff_mean = torch.empty_strided(
        (channels,),
        (1,),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    coeff_var = torch.empty_strided(
        (channels,),
        (1,),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    coeff_weight = torch.empty_strided(
        (channels,),
        (1,),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _masked_dual_sum_kernel[(channels, num_tiles)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        where,
        partial_sum,
        partial_centered_sum,
        act_stride_n=arg2_1.stride(0),
        act_stride_c=arg2_1.stride(1),
        act_stride_h=arg2_1.stride(2),
        act_stride_w=arg2_1.stride(3),
        C=channels,
        H=height,
        W=width,
        HW=hw,
        REDUCE_SIZE=reduce_size,
        NUM_TILES=num_tiles,
        BLOCK_R=BLOCK_R,
        num_warps=reduce_warps,
        num_stages=3,
    )
    _finalize_channel_kernel[(channels,)](
        partial_sum,
        partial_centered_sum,
        arg4_1,
        arg5_1,
        full,
        sum_out,
        vector_out,
        coeff_mean,
        coeff_var,
        coeff_weight,
        NUM_TILES=num_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=1,
        num_stages=3,
    )
    _bn_backward_epilogue_kernel[(triton.cdiv(numel, BLOCK_E),)](
        where,
        arg2_1,
        arg3_1,
        coeff_mean,
        coeff_var,
        coeff_weight,
        out,
        act_stride_n=arg2_1.stride(0),
        act_stride_c=arg2_1.stride(1),
        act_stride_h=arg2_1.stride(2),
        act_stride_w=arg2_1.stride(3),
        C=channels,
        H=height,
        W=width,
        HW=hw,
        NUMEL=numel,
        BLOCK_E=BLOCK_E,
        num_warps=epilogue_warps,
        num_stages=3,
    )
    return full, where, sum_out, vector_out, out
