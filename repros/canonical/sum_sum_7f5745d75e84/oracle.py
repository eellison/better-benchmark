"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 GhostNet BN-backward fragment by co-reducing the sliced bf16 add and its centered product across N*H*W into shared split-K partials, finalizing both channel sums once, and using those summaries to write the returned raw sum, scale-gradient vector, and channels-last bf16 dense epilogue, whereas Inductor lowers the sibling reductions and dependent epilogue through generic reduction scheduling; Inductor cannot do this today because its reduction codegen lacks a point-specialized cooperative split-K template that keeps equal-channel reductions and their full-tensor consumer in one coordinated plan while preserving bf16 add/output cast boundaries; the fix is COOPERATIVE_SPLIT_K: add a guarded BN-backward channel-reduction lowering that shares split-K partials for both sums and feeds the dense epilogue without redundant scheduling."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 512
IN_CHANNELS = 160
CHANNELS = 80
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW
NUMEL = ELEMENTS_PER_CHANNEL * CHANNELS
REDUCE_SCALE = 3.985969387755102e-05
OUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
OUT_STRIDE = (CHANNELS * HW, 1, CHANNELS * WIDTH, CHANNELS)
_USE_INDUCTOR_NUMERICS = False


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
def _bf16_add_to_f32(a, b):
    return _f32_add(a, b).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _partial_reduce_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    partial_sum_ptr,
    partial_prod_ptr,
    ELEMENTS_PER_CHANNEL_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    IN_CHANNELS_: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    USE_INDUCTOR_NUMERICS: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    r_offsets = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    c_mask = c_offsets < CHANNELS_
    r_mask = r_offsets < ELEMENTS_PER_CHANNEL_
    mask = r_mask[:, None] & c_mask[None, :]

    narrow_offsets = r_offsets[:, None] * CHANNELS_ + c_offsets[None, :]
    wide_offsets = r_offsets[:, None] * IN_CHANNELS_ + c_offsets[None, :]

    lhs = tl.load(arg0_ptr + wide_offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + narrow_offsets, mask=mask, other=0.0).to(tl.float32)
    if USE_INDUCTOR_NUMERICS:
        add_value = _f32_add(lhs, rhs)
    else:
        add_value = _bf16_add_to_f32(lhs, rhs)

    arg2_value = tl.load(arg2_ptr + narrow_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    mean = tl.load(arg3_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    centered = _f32_sub(arg2_value, mean[None, :])
    prod = _f32_mul(add_value, centered)

    partial_offsets = tl.program_id(1) * CHANNELS_ + c_offsets
    tl.store(partial_sum_ptr + partial_offsets, tl.sum(add_value, axis=0), mask=c_mask)
    tl.store(partial_prod_ptr + partial_offsets, tl.sum(prod, axis=0), mask=c_mask)


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_prod_ptr,
    arg4_ptr,
    arg5_ptr,
    sum_out_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    scale_grad_ptr,
    CHANNELS_: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    REDUCE_SCALE_: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    c_mask = c_offsets < CHANNELS_
    mask = (chunks[:, None] < NUM_CHUNKS) & c_mask[None, :]
    partial_offsets = chunks[:, None] * CHANNELS_ + c_offsets[None, :]

    partial_sum = tl.load(partial_sum_ptr + partial_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    partial_prod = tl.load(
        partial_prod_ptr + partial_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    sum_value = tl.sum(partial_sum, axis=0)
    prod_value = tl.sum(partial_prod, axis=0)

    invstd = tl.load(arg4_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    affine = tl.load(arg5_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    mean_term = _f32_mul(sum_value, REDUCE_SCALE_)
    prod_scaled = _f32_mul(prod_value, REDUCE_SCALE_)
    invstd_sq = _f32_mul(invstd, invstd)
    prod_coeff = _f32_mul(prod_scaled, invstd_sq)
    output_scale = _f32_mul(invstd, affine)
    scale_grad = _f32_mul(prod_value, invstd)

    tl.store(sum_out_ptr + c_offsets, sum_value, mask=c_mask)
    tl.store(mean_term_ptr + c_offsets, mean_term, mask=c_mask)
    tl.store(prod_coeff_ptr + c_offsets, prod_coeff, mask=c_mask)
    tl.store(output_scale_ptr + c_offsets, output_scale, mask=c_mask)
    tl.store(scale_grad_ptr + c_offsets, scale_grad, mask=c_mask)


@triton.jit
def _epilogue_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    out_ptr,
    NUMEL_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    IN_CHANNELS_: tl.constexpr,
    BLOCK_E: tl.constexpr,
    USE_INDUCTOR_NUMERICS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offsets < NUMEL_
    channel = offsets - (offsets // CHANNELS_) * CHANNELS_
    reduce_index = offsets // CHANNELS_
    wide_offsets = reduce_index * IN_CHANNELS_ + channel

    lhs = tl.load(arg0_ptr + wide_offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if USE_INDUCTOR_NUMERICS:
        add_value = _f32_add(lhs, rhs)
    else:
        add_value = _bf16_add_to_f32(lhs, rhs)

    arg2_value = tl.load(arg2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(arg3_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    centered = _f32_sub(arg2_value, mean)

    prod_coeff = tl.load(prod_coeff_ptr + channel, mask=mask, other=0.0).to(
        tl.float32
    )
    correction = _f32_mul(centered, prod_coeff)
    residual = _f32_sub(add_value, correction)

    mean_term = tl.load(mean_term_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    residual = _f32_sub(residual, mean_term)

    output_scale = tl.load(output_scale_ptr + channel, mask=mask, other=0.0).to(
        tl.float32
    )
    out_value = _f32_mul(residual, output_scale)
    tl.store(out_ptr + offsets, out_value.to(tl.bfloat16), mask=mask)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


# ffcd6c5b: (T([512,160,7,7], bf16, channels-last), T([512,80,7,7], bf16, channels-last), ...)
@oracle_impl(
    hardware="B200",
    point="ffcd6c5b",
    BLOCK_R=1024,
    BLOCK_C=16,
    BLOCK_E=512,
    num_warps_reduce=8,
    num_warps_final=1,
    num_warps_epilogue=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    BLOCK_E: int,
    num_warps_reduce: int,
    num_warps_final: int,
    num_warps_epilogue: int,
):
    global _USE_INDUCTOR_NUMERICS
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg0_1.device
    use_inductor_numerics = _USE_INDUCTOR_NUMERICS
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _USE_INDUCTOR_NUMERICS = True
        use_inductor_numerics = True
    num_chunks = triton.cdiv(ELEMENTS_PER_CHANNEL, BLOCK_R)
    block_chunks = _next_power_of_2(num_chunks)

    partial_sum = torch.empty_strided(
        (num_chunks, CHANNELS),
        (CHANNELS, 1),
        device=device,
        dtype=torch.float32,
    )
    partial_prod = torch.empty_strided(
        (num_chunks, CHANNELS),
        (CHANNELS, 1),
        device=device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided((CHANNELS,), (1,), device=device, dtype=torch.float32)
    mean_term = torch.empty_strided(
        (CHANNELS,), (1,), device=device, dtype=torch.float32
    )
    prod_coeff = torch.empty_strided(
        (CHANNELS,), (1,), device=device, dtype=torch.float32
    )
    output_scale = torch.empty_strided(
        (CHANNELS,), (1,), device=device, dtype=torch.float32
    )
    scale_grad = torch.empty_strided(
        (CHANNELS,), (1,), device=device, dtype=torch.float32
    )
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=device,
        dtype=torch.bfloat16,
    )

    reduce_grid = (triton.cdiv(CHANNELS, BLOCK_C), num_chunks)
    _partial_reduce_kernel[reduce_grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        partial_sum,
        partial_prod,
        ELEMENTS_PER_CHANNEL_=ELEMENTS_PER_CHANNEL,
        CHANNELS_=CHANNELS,
        IN_CHANNELS_=IN_CHANNELS,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        USE_INDUCTOR_NUMERICS=use_inductor_numerics,
        num_warps=num_warps_reduce,
    )
    _finalize_kernel[(triton.cdiv(CHANNELS, BLOCK_C),)](
        partial_sum,
        partial_prod,
        arg4_1,
        arg5_1,
        sum_out,
        mean_term,
        prod_coeff,
        output_scale,
        scale_grad,
        CHANNELS_=CHANNELS,
        NUM_CHUNKS=num_chunks,
        BLOCK_CHUNKS=block_chunks,
        BLOCK_C=BLOCK_C,
        REDUCE_SCALE_=REDUCE_SCALE,
        num_warps=num_warps_final,
    )
    _epilogue_kernel[(triton.cdiv(NUMEL, BLOCK_E),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        mean_term,
        prod_coeff,
        output_scale,
        out,
        NUMEL_=NUMEL,
        CHANNELS_=CHANNELS,
        IN_CHANNELS_=IN_CHANNELS,
        BLOCK_E=BLOCK_E,
        USE_INDUCTOR_NUMERICS=use_inductor_numerics,
        num_warps=num_warps_epilogue,
    )
    return sum_out, scale_grad, out
