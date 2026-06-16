"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle splits the GhostNet channels-last BN-backward `N,H,W` domain, uses Inductor's fused fp32 sliced-add producer for the channel sums, keeps the bf16 materialized producer for the dense return, and emits the fp32 vector plus bf16 tensor outputs; whereas Inductor schedules this sliced where producer, sibling reductions, and dependent dense epilogue as generic regions; Inductor cannot do this today because its scheduler lacks a guarded multi-output split-K lowering that preserves the mixed cast boundaries across reductions and epilogues; the fix is COOPERATIVE_SPLIT_K: add a channels-last BN-backward lowering that shares the split-K reductions and sinks finalized scalars into the vector and dense epilogues."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SCALE = 3.985969387755102e-05
EPILOGUE_BLOCK = 512


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
def _round_bf16_to_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _producer_value(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    fill_ptr,
    offset_wide,
    offset_narrow,
    mask,
):
    lhs = tl.load(arg0_ptr + offset_wide, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + offset_narrow, mask=mask, other=0.0).to(tl.float32)
    added = _f32_add(lhs, rhs)
    pred = tl.load(arg2_ptr + offset_narrow, mask=mask, other=1.0)
    fill = tl.load(fill_ptr).to(tl.float32)
    return tl.where(pred <= 0.0, fill, added)


@triton.jit
def _producer_value_rounded(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    fill_ptr,
    offset_wide,
    offset_narrow,
    mask,
):
    lhs = tl.load(arg0_ptr + offset_wide, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + offset_narrow, mask=mask, other=0.0).to(tl.float32)
    added = _round_bf16_to_f32(_f32_add(lhs, rhs))
    pred = tl.load(arg2_ptr + offset_narrow, mask=mask, other=1.0)
    fill = tl.load(fill_ptr).to(tl.float32)
    return tl.where(pred <= 0.0, fill, added)


@triton.jit
def _zero_kernel(
    sum_out_ptr,
    dot_out_ptr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < C
    tl.store(sum_out_ptr + offsets, tl.zeros((BLOCK,), dtype=tl.float32), mask=mask)
    tl.store(dot_out_ptr + offsets, tl.zeros((BLOCK,), dtype=tl.float32), mask=mask)


@triton.jit
def _atomic_reduce_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    fill_ptr,
    arg4_ptr,
    mean_ptr,
    sum_out_ptr,
    dot_out_ptr,
    C: tl.constexpr,
    C_IN: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    narrow_offsets = k_offsets[:, None] * C + c_offsets[None, :]
    wide_offsets = k_offsets[:, None] * C_IN + c_offsets[None, :]
    mask = (k_offsets[:, None] < K_TOTAL) & (c_offsets[None, :] < C)

    producer = _producer_value(
        arg0_ptr,
        arg1_ptr,
        arg2_ptr,
        fill_ptr,
        wide_offsets,
        narrow_offsets,
        mask,
    )
    activation = tl.load(arg4_ptr + narrow_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    centered = _f32_sub(activation, mean[None, :])
    active = tl.where(mask, producer, 0.0)
    active_dot = tl.where(mask, _f32_mul(producer, centered), 0.0)

    channel_mask = c_offsets < C
    tl.atomic_add(
        sum_out_ptr + c_offsets,
        tl.sum(active, axis=0),
        sem="relaxed",
        mask=channel_mask,
    )
    tl.atomic_add(
        dot_out_ptr + c_offsets,
        tl.sum(active_dot, axis=0),
        sem="relaxed",
        mask=channel_mask,
    )


@triton.jit
def _epilogue_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    fill_ptr,
    arg4_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_ptr,
    dot_ptr,
    vector_out_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    C_IN: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL
    c = offsets % C
    k = offsets // C
    wide_offsets = k * C_IN + c

    producer = _producer_value_rounded(
        arg0_ptr,
        arg1_ptr,
        arg2_ptr,
        fill_ptr,
        wide_offsets,
        offsets,
        active,
    )
    activation = tl.load(arg4_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_value = tl.load(sum_ptr + c, mask=active, other=0.0).to(tl.float32)
    dot_value = tl.load(dot_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = _f32_sub(activation, mean)
    mean_term = _f32_mul(sum_value, SCALE_VALUE)
    scaled_dot = _f32_mul(dot_value, SCALE_VALUE)
    invstd_sq = _f32_mul(invstd, invstd)
    coeff = _f32_mul(scaled_dot, invstd_sq)
    variance_term = _f32_mul(centered, coeff)
    residual = _f32_sub(producer, variance_term)
    residual = _f32_sub(residual, mean_term)
    out_scale = _f32_mul(invstd, weight)
    result = _f32_mul(residual, out_scale)
    tl.store(
        vector_out_ptr + c,
        _f32_mul(dot_value, invstd),
        mask=active & (offsets < C),
    )
    tl.store(out_ptr + offsets, result.to(tl.bfloat16), mask=active)


# 8399096d: GhostNet train, bf16 channels-last N=512, C=480, H=W=7.
@oracle_impl(hardware="B200", point="8399096d", BLOCK_K=1024, BLOCK_C=8, num_warps=4)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_C: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    n, c, h, w = arg1_1.shape
    k_total = n * h * w
    total = arg1_1.numel()
    num_k_tiles = triton.cdiv(k_total, BLOCK_K)
    c_in = int(arg0_1.shape[1])

    sum_out = torch.empty_strided((c,), (1,), device=arg1_1.device, dtype=torch.float32)
    dot_out = torch.empty_strided((c,), (1,), device=arg1_1.device, dtype=torch.float32)
    vector_out = torch.empty_strided((c,), (1,), device=arg1_1.device, dtype=torch.float32)
    out = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    _zero_kernel[(triton.cdiv(c, 1024),)](
        sum_out,
        dot_out,
        C=c,
        BLOCK=1024,
        num_warps=4,
        num_stages=3,
    )
    _atomic_reduce_kernel[(triton.cdiv(c, BLOCK_C), num_k_tiles)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        sum_out,
        dot_out,
        C=c,
        C_IN=c_in,
        K_TOTAL=k_total,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    _epilogue_kernel[(triton.cdiv(total, EPILOGUE_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        sum_out,
        dot_out,
        vector_out,
        out,
        TOTAL=total,
        C=c,
        C_IN=c_in,
        SCALE_VALUE=SCALE,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return sum_out, vector_out, out
