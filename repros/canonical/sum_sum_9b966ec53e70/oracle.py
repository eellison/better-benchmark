"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet bf16 clone/copy plus BatchNorm-backward tail by materializing the required contiguous clone and channels-last copy outputs, split-K reducing the high-channel slice into the returned f32 `sum(slice)` and scale-gradient vectors, and sinking the finalized summaries into the channels-last bf16 dense-gradient epilogue; Inductor currently schedules the memory-format clone/copy, slice producer, sibling channel reductions, vector epilogue, and dependent dense BN-backward epilogue as separate generic regions over materialized intermediates; Inductor cannot do this today because scheduler/codegen does not form one full-scope cooperative channel-reduction plan that preserves visible layout-copy outputs, bf16/f32 cast boundaries, and dependent dense epilogues together; the fix is COOPERATIVE_SPLIT_K: teach reduction scheduling to split compatible BN-backward reductions across N/H/W, combine sibling summaries once, and fuse memory-format-aware producers with downstream vector and tensor epilogues."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
INPUT_C = 160
C = 80
SLICE_START = 80
H = 7
W = 7
HW = H * W
K_TOTAL = N * HW
OUT_NUMEL = N * C * HW
SCALE = 3.985969387755102e-05
BLOCK_COPY_HW = 64
BLOCK_COPY_C = 16
BLOCK_K = 2048
BLOCK_C = 16
EPILOGUE_BLOCK = 1024


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
def _copy_visible_outputs_kernel(
    source_ptr,
    clone_out_ptr,
    copy_out_ptr,
    INPUT_C_N: tl.constexpr,
    HW_N: tl.constexpr,
    BLOCK_HW_N: tl.constexpr,
    BLOCK_C_N: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C_N + tl.arange(0, BLOCK_C_N)
    n = tl.program_id(1)
    hw_offsets = tl.arange(0, BLOCK_HW_N)
    load_offsets = n * (INPUT_C_N * HW_N) + hw_offsets[:, None] * INPUT_C_N + c_offsets[None, :]
    load_mask = (hw_offsets[:, None] < HW_N) & (c_offsets[None, :] < INPUT_C_N)
    value = tl.load(source_ptr + load_offsets, mask=load_mask, other=0.0)

    tl.store(copy_out_ptr + load_offsets, value, mask=load_mask)

    clone_offsets = n * (INPUT_C_N * HW_N) + c_offsets[:, None] * HW_N + hw_offsets[None, :]
    clone_mask = (c_offsets[:, None] < INPUT_C_N) & (hw_offsets[None, :] < HW_N)
    tl.store(clone_out_ptr + clone_offsets, tl.trans(value), mask=clone_mask)


@triton.jit
def _partial_reduce_kernel(
    grad_source_ptr,
    activation_ptr,
    mean_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    INPUT_C_N: tl.constexpr,
    C_N: tl.constexpr,
    SLICE_START_N: tl.constexpr,
    K_TOTAL_N: tl.constexpr,
    BLOCK_K_N: tl.constexpr,
    BLOCK_C_N: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C_N + tl.arange(0, BLOCK_C_N)
    k_offsets = tl.program_id(1) * BLOCK_K_N + tl.arange(0, BLOCK_K_N)
    c_mask = c_offsets < C_N
    k_mask = k_offsets < K_TOTAL_N
    mask = k_mask[:, None] & c_mask[None, :]

    grad_offsets = k_offsets[:, None] * INPUT_C_N + (SLICE_START_N + c_offsets)[None, :]
    act_offsets = k_offsets[:, None] * C_N + c_offsets[None, :]

    grad = tl.load(grad_source_ptr + grad_offsets, mask=mask, other=0.0).to(tl.float32)
    activation = tl.load(activation_ptr + act_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    centered = _f32_sub(activation, mean[None, :])

    grad = tl.where(mask, grad, 0.0)
    partial_sum = tl.sum(grad, axis=0)
    partial_dot = tl.sum(_f32_mul(grad, centered), axis=0)
    out_offsets = tl.program_id(1) * C_N + c_offsets
    tl.store(partial_sum_ptr + out_offsets, partial_sum, mask=c_mask)
    tl.store(partial_dot_ptr + out_offsets, partial_dot, mask=c_mask)


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scale_grad_out_ptr,
    stats_ptr,
    C_N: tl.constexpr,
    NUM_K_TILES_N: tl.constexpr,
    BLOCK_TILES_N: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
):
    c = tl.program_id(0)
    tile_offsets = tl.arange(0, BLOCK_TILES_N)
    mask = tile_offsets < NUM_K_TILES_N
    offsets = tile_offsets * C_N + c

    sum_values = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dot_values = tl.load(partial_dot_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_value = tl.sum(sum_values, axis=0)
    dot_value = tl.sum(dot_values, axis=0)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    mean_term = _f32_mul(sum_value, SCALE_VALUE)
    dot_mean = _f32_mul(dot_value, SCALE_VALUE)
    invstd_sq = _f32_mul(invstd, invstd)
    correction_scale = _f32_mul(dot_mean, invstd_sq)
    output_scale = _f32_mul(invstd, weight)

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(scale_grad_out_ptr + c, _f32_mul(dot_value, invstd))
    tl.store(stats_ptr + c, mean_term)
    tl.store(stats_ptr + C_N + c, correction_scale)
    tl.store(stats_ptr + 2 * C_N + c, output_scale)


@triton.jit
def _epilogue_kernel(
    grad_source_ptr,
    activation_ptr,
    mean_ptr,
    stats_ptr,
    dense_out_ptr,
    INPUT_C_N: tl.constexpr,
    C_N: tl.constexpr,
    SLICE_START_N: tl.constexpr,
    OUT_NUMEL_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < OUT_NUMEL_N
    c = offsets % C_N
    k = offsets // C_N
    grad_offsets = k * INPUT_C_N + SLICE_START_N + c

    grad = tl.load(grad_source_ptr + grad_offsets, mask=mask, other=0.0).to(tl.float32)
    activation = tl.load(activation_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    mean_term = tl.load(stats_ptr + c, mask=mask, other=0.0).to(tl.float32)
    correction_scale = tl.load(stats_ptr + C_N + c, mask=mask, other=0.0).to(tl.float32)
    output_scale = tl.load(stats_ptr + 2 * C_N + c, mask=mask, other=0.0).to(tl.float32)

    centered = _f32_sub(activation, mean)
    correction = _f32_mul(centered, correction_scale)
    after_correction = _f32_sub(grad, correction)
    after_mean = _f32_sub(after_correction, mean_term)
    result = _f32_mul(after_mean, output_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(dense_out_ptr + offsets, result, mask=mask)


# 3daf9266: GhostNet train, N=512, high slice C=80 of bf16 channels-last [512,160,7,7].
@oracle_impl(hardware="B200", point="3daf9266")
def oracle_forward(inputs):
    source, activation, mean, invstd, weight = inputs
    num_k_tiles = triton.cdiv(K_TOTAL, BLOCK_K)
    block_tiles = _next_power_of_2(num_k_tiles)

    clone_out = torch.empty_strided(
        (N, INPUT_C, H, W),
        (INPUT_C * HW, HW, W, 1),
        device=source.device,
        dtype=torch.bfloat16,
    )
    copy_out = torch.empty_strided(
        (N, INPUT_C, H, W),
        (INPUT_C * HW, 1, W * INPUT_C, INPUT_C),
        device=source.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided((C,), (1,), device=source.device, dtype=torch.float32)
    scale_grad_out = torch.empty_strided((C,), (1,), device=source.device, dtype=torch.float32)
    stats = torch.empty((3, C), device=source.device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=source.device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty((num_k_tiles, C), device=source.device, dtype=torch.float32)
    partial_dot = torch.empty_like(partial_sum)

    _copy_visible_outputs_kernel[(triton.cdiv(INPUT_C, BLOCK_COPY_C), N)](
        source,
        clone_out,
        copy_out,
        INPUT_C_N=INPUT_C,
        HW_N=HW,
        BLOCK_HW_N=BLOCK_COPY_HW,
        BLOCK_C_N=BLOCK_COPY_C,
        num_warps=4,
    )
    _partial_reduce_kernel[(triton.cdiv(C, BLOCK_C), num_k_tiles)](
        source,
        activation,
        mean,
        partial_sum,
        partial_dot,
        INPUT_C_N=INPUT_C,
        C_N=C,
        SLICE_START_N=SLICE_START,
        K_TOTAL_N=K_TOTAL,
        BLOCK_K_N=BLOCK_K,
        BLOCK_C_N=BLOCK_C,
        num_warps=8,
    )
    _finalize_kernel[(C,)](
        partial_sum,
        partial_dot,
        invstd,
        weight,
        sum_out,
        scale_grad_out,
        stats,
        C_N=C,
        NUM_K_TILES_N=num_k_tiles,
        BLOCK_TILES_N=block_tiles,
        SCALE_VALUE=SCALE,
        num_warps=8,
    )
    _epilogue_kernel[(triton.cdiv(OUT_NUMEL, EPILOGUE_BLOCK),)](
        source,
        activation,
        mean,
        stats,
        dense_out,
        INPUT_C_N=INPUT_C,
        C_N=C,
        SLICE_START_N=SLICE_START,
        OUT_NUMEL_N=OUT_NUMEL,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
    )
    return clone_out, copy_out, sum_out, scale_grad_out, dense_out
