"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle preserves the full phlippe DenseNet BN-backward return scope by cooperatively split-K reducing the two shared per-channel f32 summaries over `(N,H,W)`, finalizing `sum(where)` and `sum(where * centered) * invstd`, and sinking those summaries into the dependent bf16 gradient/add epilogue while returning the required aliasing `add[:, :16]` view. Inductor currently schedules the bf16 masked producer, sibling reductions, broadcast BN-backward epilogue, slice add, and returned aliasing slice as separate generic regions around materialized intermediates; it cannot do this today because scheduler/codegen does not form a multi-output channel split-K plan that reuses the same masked/centered producer and finalized summaries while preserving bf16 cast boundaries and output aliasing. The fix is COOPERATIVE_SPLIT_K: teach reduction scheduling to split compatible BN-backward channel reductions over the reduced domain, combine partials once, and fuse the dependent dense/add epilogue with exact dtype boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SCALE = 3.0517578125e-05
SLICE_START = 16
EPILOGUE_BLOCK_C = 8


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
def _partial_reduce_kernel(
    mask_ptr,
    fill_ptr,
    rhs_ptr,
    activation_ptr,
    mean_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    n_offsets = k_offsets // HW
    hw_offsets = k_offsets - n_offsets * HW
    offsets = n_offsets[:, None] * (C * HW) + c_offsets[None, :] * HW + hw_offsets[:, None]
    mask = (k_offsets[:, None] < K_TOTAL) & (c_offsets[None, :] < C)

    gate = tl.load(mask_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)
    selected = tl.where(gate <= 0.0, fill, rhs)

    activation = tl.load(activation_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    centered = _f32_sub(activation, mean[None, :])
    dot = _f32_mul(selected, centered)

    partial_sum = tl.sum(tl.where(mask, selected, 0.0), axis=0)
    partial_dot = tl.sum(tl.where(mask, dot, 0.0), axis=0)
    out_offsets = tl.program_id(1) * C + c_offsets
    tl.store(partial_sum_ptr + out_offsets, partial_sum, mask=c_offsets < C)
    tl.store(partial_dot_ptr + out_offsets, partial_dot, mask=c_offsets < C)


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scaled_dot_out_ptr,
    mean_term_ptr,
    correction_scale_ptr,
    output_scale_ptr,
    C: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
):
    c = tl.program_id(0)
    tile_offsets = tl.arange(0, BLOCK_TILES)
    mask = tile_offsets < NUM_K_TILES
    offsets = tile_offsets * C + c

    sum_value = tl.sum(
        tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    dot_value = tl.sum(
        tl.load(partial_dot_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    invstd_sq = _f32_mul(invstd, invstd)
    dot_mean = _f32_mul(dot_value, SCALE_VALUE)
    correction_scale = _f32_mul(dot_mean, invstd_sq)
    output_scale = _f32_mul(invstd, weight)

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(scaled_dot_out_ptr + c, _f32_mul(dot_value, invstd))
    tl.store(mean_term_ptr + c, _f32_mul(sum_value, SCALE_VALUE))
    tl.store(correction_scale_ptr + c, correction_scale)
    tl.store(output_scale_ptr + c, output_scale)


@triton.jit
def _epilogue_kernel(
    sliced_ptr,
    mask_ptr,
    fill_ptr,
    rhs_ptr,
    activation_ptr,
    mean_ptr,
    mean_term_ptr,
    correction_scale_ptr,
    output_scale_ptr,
    out_ptr,
    C: tl.constexpr,
    INPUT_C: tl.constexpr,
    SLICE_OFFSET: tl.constexpr,
    HW: tl.constexpr,
    NUM_HW_TILES: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    spatial_pid = tl.program_id(1)
    n = spatial_pid // NUM_HW_TILES
    hw_tile = spatial_pid - n * NUM_HW_TILES
    hw = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c_matrix = c[:, None]
    hw_matrix = hw[None, :]
    active = (c_matrix < C) & (hw_matrix < HW)
    compact_offsets = n * (C * HW) + c_matrix * HW + hw_matrix
    slice_offsets = n * (INPUT_C * HW) + (c_matrix + SLICE_OFFSET) * HW + hw_matrix

    gate = tl.load(mask_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)
    selected = tl.where(gate <= 0.0, fill, rhs)

    activation = tl.load(activation_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    c_mask = c < C
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    centered = _f32_sub(activation, mean)
    correction_scale = tl.load(correction_scale_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    correction = _f32_mul(centered, correction_scale)
    after_correction = _f32_sub(selected, correction)
    mean_term = tl.load(mean_term_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    centered_grad = _f32_sub(after_correction, mean_term)
    output_scale = tl.load(output_scale_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    grad = _f32_mul(centered_grad, output_scale)
    grad_bf16 = grad.to(tl.bfloat16, fp_downcast_rounding="rtne")

    sliced = tl.load(sliced_ptr + slice_offsets, mask=active, other=0.0).to(tl.float32)
    add_value = _f32_add(sliced, grad_bf16.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(out_ptr + compact_offsets, add_value, mask=active)


def _launch(inputs, *, BLOCK_K: int, BLOCK_C: int, BLOCK_HW: int, num_warps: int):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7 = inputs
    n, c, h, w = arg1.shape
    hw = h * w
    k_total = n * hw
    input_c = int(arg0.shape[1])
    num_hw_tiles = triton.cdiv(hw, BLOCK_HW)
    num_k_tiles = triton.cdiv(k_total, BLOCK_K)
    block_tiles = _next_power_of_2(num_k_tiles)

    partial_sum = torch.empty((num_k_tiles, c), device=arg1.device, dtype=torch.float32)
    partial_dot = torch.empty((num_k_tiles, c), device=arg1.device, dtype=torch.float32)
    sum_out = torch.empty((c,), device=arg1.device, dtype=torch.float32)
    scaled_dot_out = torch.empty((c,), device=arg1.device, dtype=torch.float32)
    mean_term = torch.empty((c,), device=arg1.device, dtype=torch.float32)
    correction_scale = torch.empty((c,), device=arg1.device, dtype=torch.float32)
    output_scale = torch.empty((c,), device=arg1.device, dtype=torch.float32)
    add_out = torch.empty_strided(
        (n, c, h, w),
        (c * hw, hw, w, 1),
        device=arg1.device,
        dtype=torch.bfloat16,
    )

    _partial_reduce_kernel[(triton.cdiv(c, BLOCK_C), num_k_tiles)](
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        partial_sum,
        partial_dot,
        C=c,
        HW=hw,
        K_TOTAL=k_total,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    _finalize_kernel[(c,)](
        partial_sum,
        partial_dot,
        arg6,
        arg7,
        sum_out,
        scaled_dot_out,
        mean_term,
        correction_scale,
        output_scale,
        C=c,
        NUM_K_TILES=num_k_tiles,
        BLOCK_TILES=block_tiles,
        SCALE_VALUE=SCALE,
        num_warps=8,
    )
    _epilogue_kernel[(triton.cdiv(c, EPILOGUE_BLOCK_C), n * num_hw_tiles)](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        mean_term,
        correction_scale,
        output_scale,
        add_out,
        C=c,
        INPUT_C=input_c,
        SLICE_OFFSET=SLICE_START,
        HW=hw,
        NUM_HW_TILES=num_hw_tiles,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=EPILOGUE_BLOCK_C,
        num_warps=4,
    )
    return sum_out, scaled_dot_out, add_out, add_out[:, :SLICE_START, :, :]


@oracle_impl(hardware="B200", point="4363851f", BLOCK_K=1024, BLOCK_C=8, BLOCK_HW=256, num_warps=4)
@oracle_impl(hardware="B200", point="71c373f0", BLOCK_K=1024, BLOCK_C=8, BLOCK_HW=64, num_warps=4)
def oracle_forward(inputs, **kwargs):
    return _launch(inputs, **kwargs)
