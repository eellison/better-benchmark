"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 Visformer BN-backward/ReLU-backward tail by split-K reducing the shared channels-last masked producer `where(arg0 <= 0, 0, arg1)` into both returned f32 channel summaries, then using the finalized summaries to write the returned channels-last bf16 dense gradient. Inductor currently schedules the bf16 mask producer, sibling `sum([0, 2, 3])` reductions, scale-gradient vector, and reduction-dependent dense epilogue as generic regions around repeated producer work; it cannot form one full-scope split-K plan that preserves the bf16 input/output cast boundaries and keeps the channel summaries live for the dense epilogue. The fix is COOPERATIVE_SPLIT_K: add a guarded channels-last BN-backward split-K template that shares compatible channel reductions and sinks the vector and dense epilogues into one coordinated schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 32
H = 112
W = 112
HW = H * W
K_TOTAL = N * HW
TOTAL = K_TOTAL * C
SCALE = 6.228077168367346e-07


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


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
def _masked_producer(mask_bf16, source_bf16):
    mask_value = mask_bf16.to(tl.float32)
    source_value = source_bf16.to(tl.float32)
    return tl.where(mask_value <= 0.0, 0.0, source_value)


@triton.jit
def _partial_reduce_kernel(
    mask_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    NUM_K_BLOCKS: tl.constexpr,
    K_TOTAL_: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    active = (k[:, None] < K_TOTAL_) & (c[None, :] < C_)
    offsets = k[:, None] * C_ + c[None, :]

    mask_bf16 = tl.load(mask_ptr + offsets, mask=active, other=0.0)
    source_bf16 = tl.load(source_ptr + offsets, mask=active, other=0.0)
    centered_source = tl.load(centered_source_ptr + offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c, mask=c < C_, other=0.0).to(tl.float32)

    producer = _masked_producer(mask_bf16, source_bf16)
    centered = _f32_sub(centered_source, mean[None, :])
    producer = tl.where(active, producer, 0.0)
    dot = tl.where(active, _f32_mul(producer, centered), 0.0)

    partial_offsets = c * NUM_K_BLOCKS + tl.program_id(1)
    tl.store(
        partial_sum_ptr + partial_offsets,
        tl.sum(producer, axis=0),
        mask=c < C_,
    )
    tl.store(
        partial_dot_ptr + partial_offsets,
        tl.sum(dot, axis=0),
        mask=c < C_,
    )


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    sum_out_ptr,
    dot_tmp_ptr,
    scaled_dot_ptr,
    NUM_K_BLOCKS: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
):
    c = tl.program_id(0)
    block_offsets = tl.arange(0, BLOCK_BLOCKS)
    active = block_offsets < NUM_K_BLOCKS
    offsets = c * NUM_K_BLOCKS + block_offsets

    sum_value = tl.sum(
        tl.load(partial_sum_ptr + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )
    dot_value = tl.sum(
        tl.load(partial_dot_ptr + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )
    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(dot_tmp_ptr + c, dot_value)
    tl.store(scaled_dot_ptr + c, _f32_mul(dot_value, invstd))


@triton.jit
def _epilogue_kernel(
    mask_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_ptr,
    dot_ptr,
    out_ptr,
    TOTAL_: tl.constexpr,
    C_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL_
    c = offsets % C_

    mask_bf16 = tl.load(mask_ptr + offsets, mask=active, other=0.0)
    source_bf16 = tl.load(source_ptr + offsets, mask=active, other=0.0)
    centered_source = tl.load(centered_source_ptr + offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_value = tl.load(sum_ptr + c, mask=active, other=0.0).to(tl.float32)
    dot_value = tl.load(dot_ptr + c, mask=active, other=0.0).to(tl.float32)

    producer = _masked_producer(mask_bf16, source_bf16)
    centered = _f32_sub(centered_source, mean)
    mean_term = _f32_mul(sum_value, SCALE_)
    dot_scaled = _f32_mul(dot_value, SCALE_)
    invstd_sq = _f32_mul(invstd, invstd)
    variance_term = _f32_mul(dot_scaled, invstd_sq)
    output_scale = _f32_mul(invstd, weight)
    after_variance = _f32_sub(producer, _f32_mul(centered, variance_term))
    after_mean = _f32_sub(after_variance, mean_term)
    out = _f32_mul(after_mean, output_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(out_ptr + offsets, out, mask=active)


@oracle_impl(
    hardware="B200",
    point="bf7e56a6",
    BLOCK_K=1024,
    BLOCK_C=8,
    EPILOGUE_BLOCK=1024,
    reduce_warps=8,
    final_warps=8,
    epilogue_warps=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    BLOCK_C: int,
    EPILOGUE_BLOCK: int,
    reduce_warps: int,
    final_warps: int,
    epilogue_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    num_k_blocks = triton.cdiv(K_TOTAL, BLOCK_K)

    sum_out = torch.empty_strided((C,), (1,), device=arg1_1.device, dtype=torch.float32)
    dot_tmp = torch.empty_strided((C,), (1,), device=arg1_1.device, dtype=torch.float32)
    scaled_dot = torch.empty_strided(
        (C,), (1,), device=arg1_1.device, dtype=torch.float32
    )
    partial_sum = torch.empty_strided(
        (C, num_k_blocks),
        (num_k_blocks, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    partial_dot = torch.empty_like(partial_sum)
    out = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    _partial_reduce_kernel[(triton.cdiv(C, BLOCK_C), num_k_blocks)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        partial_sum,
        partial_dot,
        NUM_K_BLOCKS=num_k_blocks,
        K_TOTAL_=K_TOTAL,
        C_=C,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=reduce_warps,
        num_stages=3,
    )
    _finalize_kernel[(C,)](
        partial_sum,
        partial_dot,
        arg4_1,
        sum_out,
        dot_tmp,
        scaled_dot,
        NUM_K_BLOCKS=num_k_blocks,
        BLOCK_BLOCKS=_next_power_of_2(num_k_blocks),
        num_warps=final_warps,
        num_stages=3,
    )
    _epilogue_kernel[(triton.cdiv(TOTAL, EPILOGUE_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        sum_out,
        dot_tmp,
        out,
        TOTAL_=TOTAL,
        C_=C,
        SCALE_=SCALE,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=epilogue_warps,
        num_stages=4,
    )
    return sum_out, scaled_dot, out
