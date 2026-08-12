"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ShuffleNet bf16 channel-shuffle plus BN-backward tail, including the returned contiguous shuffled tensor, the bf16-rounded ReLU mask feeding the `where`, both f32 per-channel reductions, the scale-gradient vector, and the final channels-last bf16 dense gradient. Inductor currently lowers the channel-shuffle materialization, masked producer, sibling reductions, and reduction-dependent broadcast epilogue as separate generic schedules around materialized intermediates; it cannot do this today because scheduler/codegen does not represent the fixed channel shuffle as a virtual producer feeding a multi-output channel reduction with a dependent dense epilogue while preserving bf16 cast boundaries. The fix is SCHEDULER_FUSION: teach reduction scheduling to co-schedule compatible BN-backward reductions, sink static layout producers, and emit the dependent tensor/vector epilogues in one full-scope plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 232
OUT_C = 464
H = 7
W = 7
HW = H * W
R = N * HW
SHUFFLE_NUMEL = N * OUT_C * HW
BN_NUMEL = N * C * HW
SCALE = 0.00015943877551020407


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
def _shuffle_materialize_kernel(
    source_ptr,
    shuffled_out_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL

    w = offsets % 7
    h = (offsets // 7) % 7
    out_c = (offsets // 49) % 464
    n = offsets // 22736

    src_c = tl.where(out_c < 232, out_c * 2, (out_c - 232) * 2 + 1)
    source_offsets = n * 22736 + src_c + h * 3248 + w * 464
    values = tl.load(source_ptr + source_offsets, mask=active, other=0.0)
    tl.store(shuffled_out_ptr + offsets, values, mask=active)


@triton.jit
def _bn_reduce_kernel(
    shuffle_source_ptr,
    bn_input_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    sum_out_ptr,
    sum_centered_ptr,
    R_SIZE: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    active = rows < R_SIZE

    n = rows // 49
    spatial = rows - n * 49
    h = spatial // 7
    w = spatial - h * 7

    bn_offsets = n * 11368 + c + h * 1624 + w * 232
    shuffle_offsets = n * 22736 + (c * 2 + 1) + h * 3248 + w * 464

    bn_input = tl.load(bn_input_ptr + bn_offsets, mask=active, other=0.0).to(tl.float32)
    source_value = tl.load(
        shuffle_source_ptr + shuffle_offsets, mask=active, other=0.0
    )
    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    fill_value = tl.load(fill_ptr)

    centered = _f32_sub(bn_input, mean)
    normalized = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    where_bf16 = tl.where(affine_bf16.to(tl.float32) <= 0.0, fill_value, source_value)
    where_f32 = tl.where(active, where_bf16.to(tl.float32), 0.0)
    centered = tl.where(active, centered, 0.0)

    product = _f32_mul(where_f32, centered)
    sum_where = tl.sum(where_f32, axis=0)
    sum_centered = tl.sum(product, axis=0)

    tl.store(sum_out_ptr + c, sum_where)
    tl.store(sum_centered_ptr + c, sum_centered)


@triton.jit
def _bn_epilogue_kernel(
    shuffle_source_ptr,
    bn_input_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    sum_out_ptr,
    sum_centered_ptr,
    scale_grad_ptr,
    dense_out_ptr,
    TOTAL: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < TOTAL

    c = linear % 232
    w = (linear // 232) % 7
    h = (linear // 1624) % 7
    n = linear // 11368

    shuffle_offsets = n * 22736 + (c * 2 + 1) + h * 3248 + w * 464
    bn_input = tl.load(bn_input_ptr + linear, mask=active, other=0.0).to(tl.float32)
    source_value = tl.load(
        shuffle_source_ptr + shuffle_offsets, mask=active, other=0.0
    )
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=active, other=0.0).to(tl.float32)
    fill_value = tl.load(fill_ptr)
    sum_where = tl.load(sum_out_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_centered = tl.load(sum_centered_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = _f32_sub(bn_input, mean)
    normalized = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    where_bf16 = tl.where(affine_bf16.to(tl.float32) <= 0.0, fill_value, source_value)
    where_f32 = where_bf16.to(tl.float32)

    mean_term = _f32_mul(sum_where, SCALE_VALUE)
    dot_scaled = _f32_mul(sum_centered, SCALE_VALUE)
    invstd_sq = _f32_mul(invstd, invstd)
    variance_term = _f32_mul(dot_scaled, invstd_sq)
    output_scale = _f32_mul(invstd, weight)

    after_variance = _f32_sub(where_f32, _f32_mul(centered, variance_term))
    after_mean = _f32_sub(after_variance, mean_term)
    dense_bf16 = _f32_mul(after_mean, output_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(dense_out_ptr + linear, dense_bf16, mask=active)
    tl.store(
        scale_grad_ptr + c,
        _f32_mul(sum_centered, invstd),
        mask=active & (linear < 232),
    )


# 0e23e8e0: ShuffleNet train, bf16 channels-last BN input with observable channel shuffle.
@oracle_impl(
    hardware="B200",
    point="0e23e8e0",
    SHUFFLE_BLOCK=1024,
    BLOCK_R=8192,
    EPILOGUE_BLOCK=256,
    shuffle_warps=4,
    reduce_warps=8,
    epilogue_warps=4,
)
def oracle_forward(
    inputs,
    *,
    SHUFFLE_BLOCK: int,
    BLOCK_R: int,
    EPILOGUE_BLOCK: int,
    shuffle_warps: int,
    reduce_warps: int,
    epilogue_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, _, _ = inputs
    device = arg0_1.device

    shuffled = torch.empty_strided(
        (N, OUT_C, H, W),
        (OUT_C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    sum_centered = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device,
        dtype=torch.bfloat16,
    )

    _shuffle_materialize_kernel[(triton.cdiv(SHUFFLE_NUMEL, SHUFFLE_BLOCK),)](
        arg0_1,
        shuffled,
        TOTAL=SHUFFLE_NUMEL,
        BLOCK=SHUFFLE_BLOCK,
        num_warps=shuffle_warps,
    )
    _bn_reduce_kernel[(C,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        sum_out,
        sum_centered,
        R_SIZE=R,
        BLOCK_R=BLOCK_R,
        num_warps=reduce_warps,
        num_stages=3,
    )
    _bn_epilogue_kernel[(triton.cdiv(BN_NUMEL, EPILOGUE_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        sum_out,
        sum_centered,
        scale_grad,
        dense_out,
        TOTAL=BN_NUMEL,
        SCALE_VALUE=SCALE,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=epilogue_warps,
    )

    return shuffled, sum_out, scale_grad, dense_out
