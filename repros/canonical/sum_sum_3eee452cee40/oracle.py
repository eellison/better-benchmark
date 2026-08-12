"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 MobileNetV3 hard-swish batchnorm-backward fragment by preserving the two explicit bf16 round-trips, sharing the rounded hard-swish-gradient producer across both channel sums, and reusing those finalized sums for the returned bf16 input-gradient tensor and f32 side vector, whereas Inductor schedules the broadcast producer, sibling `sum([0, 2, 3])` reductions, and dependent epilogue as generic reduction/pointwise work that rereads and recomputes large channels-last tensors; Inductor cannot do this today because its scheduler/codegen lacks a full-scope multi-output channel-reduction template that coordinates compatible sibling reductions with required bf16 cast boundaries and dependent tensor/vector epilogues; the fix is SCHEDULER_FUSION: add a channels-last hard-swish BN-backward reduction template that keeps the shared producer and both post-reduction epilogues in one planned lowering."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


CAPTURED_INV_NHW = 2.4912308673469386e-06


@triton.jit
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_f32(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _partial_sums_kernel(
    arg0_ptr,
    arg1_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    full_ptr,
    partial_sum_ptr,
    partial_centered_ptr,
    K: tl.constexpr,
    C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    offsets = k_offsets[:, None] * C + c_offsets[None, :]
    mask = (k_offsets[:, None] < K) & (c_offsets[None, :] < C)

    x = tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    activation = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)

    centered = _sub_f32(activation, mean[None, :])
    affine = _mul_f32(centered, invstd[None, :])
    affine = _mul_f32(affine, weight[None, :])
    affine = _add_f32(affine, bias[None, :])
    affine_bf16 = _round_to_bf16_f32(affine)

    middle = _mul_f32(x, _add_f32(affine_bf16 / 3.0, 0.5))
    grad = tl.where(affine_bf16 < 3.0, middle, x)
    grad = tl.where(affine_bf16 <= -3.0, full_value, grad)
    rounded_grad = _round_to_bf16_f32(grad)
    rounded_grad = tl.where(mask, rounded_grad, 0.0)

    partial_offset = tl.program_id(1) * C + c_offsets
    centered_product = _mul_f32(rounded_grad, centered)
    tl.store(
        partial_sum_ptr + partial_offset,
        tl.sum(rounded_grad, axis=0),
        mask=c_offsets < C,
    )
    tl.store(
        partial_centered_ptr + partial_offset,
        tl.sum(centered_product, axis=0),
        mask=c_offsets < C,
    )


@triton.jit
def _finalize_sums_kernel(
    partial_sum_ptr,
    partial_centered_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    mul_out_ptr,
    stats_ptr,
    C: tl.constexpr,
    NUM_TILES: tl.constexpr,
    INV_K: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tile_offsets = tl.arange(0, BLOCK_TILES)
    mask = tile_offsets < NUM_TILES
    offsets = tile_offsets * C + c

    sum_vals = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    centered_vals = tl.load(partial_centered_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_grad = tl.sum(sum_vals, axis=0)
    sum_centered = tl.sum(centered_vals, axis=0)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    mean_term = sum_grad * INV_K
    invstd_sq = invstd * invstd
    centered_term = (sum_centered * INV_K) * invstd_sq
    output_scale = invstd * weight

    tl.store(sum_out_ptr + c, sum_grad)
    tl.store(mul_out_ptr + c, sum_centered * invstd)
    tl.store(stats_ptr + c, mean_term)
    tl.store(stats_ptr + C + c, centered_term)
    tl.store(stats_ptr + 2 * C + c, output_scale)


@triton.jit
def _epilogue_kernel(
    arg0_ptr,
    arg1_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    full_ptr,
    stats_ptr,
    out_ptr,
    NUMEL: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < NUMEL
    c = offsets % C

    x = tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    activation = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)

    centered = _sub_f32(activation, mean)
    affine = _mul_f32(centered, invstd)
    affine = _mul_f32(affine, weight)
    affine = _add_f32(affine, bias)
    affine_bf16 = _round_to_bf16_f32(affine)

    middle = _mul_f32(x, _add_f32(affine_bf16 / 3.0, 0.5))
    grad = tl.where(affine_bf16 < 3.0, middle, x)
    grad = tl.where(affine_bf16 <= -3.0, full_value, grad)
    rounded_grad = _round_to_bf16_f32(grad)

    mean_term = tl.load(stats_ptr + c, mask=mask, other=0.0).to(tl.float32)
    centered_term = tl.load(stats_ptr + C + c, mask=mask, other=0.0).to(tl.float32)
    output_scale = tl.load(stats_ptr + 2 * C + c, mask=mask, other=0.0).to(tl.float32)
    out = _sub_f32(rounded_grad, _mul_f32(centered, centered_term))
    out = _sub_f32(out, mean_term)
    out = _mul_f32(out, output_scale)
    tl.store(out_ptr + offsets, out.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


def _channels_last_stride(n: int, c: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (c * h * w, 1, c * w, c)


def _run_oracle(inputs, *, BLOCK_K: int, BLOCK_C: int, EPILOGUE_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    k = n * h * w
    numel = k * c
    num_tiles = triton.cdiv(k, BLOCK_K)
    block_tiles = triton.next_power_of_2(num_tiles)

    partial_sum = torch.empty((num_tiles, c), device=arg0_1.device, dtype=torch.float32)
    partial_centered = torch.empty((num_tiles, c), device=arg0_1.device, dtype=torch.float32)
    sum_out = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)
    mul_out = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)
    stats = torch.empty((3, c), device=arg0_1.device, dtype=torch.float32)
    out = torch.empty_strided(
        (n, c, h, w),
        _channels_last_stride(n, c, h, w),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid_reduce = (triton.cdiv(c, BLOCK_C), num_tiles)
    _partial_sums_kernel[grid_reduce](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        partial_sum,
        partial_centered,
        K=k,
        C=c,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=8,
    )

    _finalize_sums_kernel[(c,)](
        partial_sum,
        partial_centered,
        arg3_1,
        arg4_1,
        sum_out,
        mul_out,
        stats,
        C=c,
        NUM_TILES=num_tiles,
        INV_K=CAPTURED_INV_NHW,
        BLOCK_TILES=block_tiles,
        num_warps=8,
    )

    _epilogue_kernel[(triton.cdiv(numel, EPILOGUE_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        stats,
        out,
        NUMEL=numel,
        C=c,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
    )
    return sum_out, mul_out, out


# de9c358f: (T([512,240,28,28], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="de9c358f", BLOCK_K=256, BLOCK_C=8, EPILOGUE_BLOCK=256)
# 1a917d0f: (T([512,240,14,14], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="1a917d0f", BLOCK_K=256, BLOCK_C=8, EPILOGUE_BLOCK=256)
# 2dd96726: (T([512,200,14,14], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="2dd96726", BLOCK_K=256, BLOCK_C=8, EPILOGUE_BLOCK=256)
# 9f0866a3: (T([512,184,14,14], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="9f0866a3", BLOCK_K=256, BLOCK_C=8, EPILOGUE_BLOCK=256)
# 3e13b6db: (T([512,480,14,14], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="3e13b6db", BLOCK_K=256, BLOCK_C=8, EPILOGUE_BLOCK=256)
# bcf5ae8d: (T([512,672,14,14], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="bcf5ae8d", BLOCK_K=256, BLOCK_C=8, EPILOGUE_BLOCK=256)
# 843d4a4d: (T([512,960,7,7], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="843d4a4d", BLOCK_K=256, BLOCK_C=8, EPILOGUE_BLOCK=256)
# e9945fee: (T([32,240,28,28], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="e9945fee", BLOCK_K=256, BLOCK_C=8, EPILOGUE_BLOCK=256)
# 90e96d75: (T([32,240,14,14], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="90e96d75", BLOCK_K=256, BLOCK_C=8, EPILOGUE_BLOCK=256)
# fab0dcaa: (T([32,200,14,14], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="fab0dcaa", BLOCK_K=256, BLOCK_C=8, EPILOGUE_BLOCK=256)
# ec0b4267: (T([32,184,14,14], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="ec0b4267", BLOCK_K=256, BLOCK_C=8, EPILOGUE_BLOCK=256)
# d0c4278a: (T([32,480,14,14], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="d0c4278a", BLOCK_K=256, BLOCK_C=8, EPILOGUE_BLOCK=256)
# 9542f2d8: (T([32,672,14,14], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="9542f2d8", BLOCK_K=256, BLOCK_C=8, EPILOGUE_BLOCK=256)
# 9b1311e4: (T([32,960,7,7], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="9b1311e4", BLOCK_K=256, BLOCK_C=8, EPILOGUE_BLOCK=256)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_C: int, EPILOGUE_BLOCK: int):
    return _run_oracle(
        inputs,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        EPILOGUE_BLOCK=EPILOGUE_BLOCK,
    )
