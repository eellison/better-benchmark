"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 batch-norm-backward tail from `Repro.forward`, sharing the masked bf16 `where` producer across both channel reductions, materializing the returned `sum(where)` and `sum(where * centered) * weight` vectors, and then using those summaries in the visible full bf16 epilogue plus the `[:, :16, :, :]` slice view, whereas Inductor schedules the masked producer, sibling reductions, broadcast arithmetic, bf16 cast, and slice output as generic separated work over the dense tensors; Inductor cannot do this today because scheduler fusion does not form one full-scope multi-output reduction feeding a dense epilogue while preserving the returned full tensor and aliasing slice; the fix is SCHEDULER_FUSION: extend the normalization-backward scheduler to fuse the common producer into the paired reductions and sink the exact broadcast epilogue and slice metadata into the same plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


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
def _partial_sums_kernel(
    mask_input_ptr,
    fill_ptr,
    where_rhs_ptr,
    centered_base_ptr,
    mean_ptr,
    partial_where_ptr,
    partial_where_centered_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    mean_stride_c: tl.constexpr,
    BLOCK_E: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_block = tl.program_id(0)
    e_block = tl.program_id(1)
    channels = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    e_offsets = e_block * BLOCK_E + tl.arange(0, BLOCK_E)

    hw = H * W
    hw_idx = e_offsets % hw
    n_idx = e_offsets // hw
    h_idx = hw_idx // W
    w_idx = hw_idx - h_idx * W
    offsets = (
        n_idx[None, :] * stride_n
        + channels[:, None] * stride_c
        + h_idx[None, :] * stride_h
        + w_idx[None, :] * stride_w
    )
    mask = (channels[:, None] < C) & (e_offsets[None, :] < E)

    mask_input = tl.load(mask_input_ptr + offsets, mask=mask, other=0.0)
    rhs = tl.load(where_rhs_ptr + offsets, mask=mask, other=0.0)
    fill = tl.load(fill_ptr + 0)
    where_bf16 = tl.where(mask_input <= 0.0, fill, rhs)
    where_f32 = tl.where(mask, where_bf16.to(tl.float32), 0.0)

    base = tl.load(centered_base_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels * mean_stride_c, mask=channels < C, other=0.0).to(
        tl.float32
    )
    centered = tl.where(mask, _f32_sub(base, mean[:, None]), 0.0)
    prod = _f32_mul(where_f32, centered)

    partial_offsets = e_block * C + channels
    channel_mask = channels < C
    tl.store(partial_where_ptr + partial_offsets, tl.sum(where_f32, axis=1), mask=channel_mask)
    tl.store(
        partial_where_centered_ptr + partial_offsets,
        tl.sum(prod, axis=1),
        mask=channel_mask,
    )


@triton.jit
def _finalize_sums_kernel(
    partial_where_ptr,
    partial_where_centered_ptr,
    weight_ptr,
    sum_where_ptr,
    grad_weight_ptr,
    sum_centered_tmp_ptr,
    C: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_block = tl.program_id(0)
    channels = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (channels[:, None] < C) & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + channels[:, None]

    sum_where = tl.sum(tl.load(partial_where_ptr + offsets, mask=mask, other=0.0), axis=1)
    sum_centered = tl.sum(
        tl.load(partial_where_centered_ptr + offsets, mask=mask, other=0.0),
        axis=1,
    )
    channel_mask = channels < C
    weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)

    tl.store(sum_where_ptr + channels, sum_where, mask=channel_mask)
    tl.store(sum_centered_tmp_ptr + channels, sum_centered, mask=channel_mask)
    tl.store(grad_weight_ptr + channels, _f32_mul(sum_centered, weight), mask=channel_mask)


@triton.jit
def _dense_epilogue_kernel(
    mask_input_ptr,
    fill_ptr,
    where_rhs_ptr,
    centered_base_ptr,
    mean_ptr,
    weight_ptr,
    affine_rhs_ptr,
    sum_where_ptr,
    sum_centered_ptr,
    out_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    TOTAL: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    mean_stride_c: tl.constexpr,
    BLOCK: tl.constexpr,
):
    out_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = out_offsets < TOTAL

    hw = H * W
    n_idx = out_offsets // (C * hw)
    channel = (out_offsets // hw) % C
    hw_idx = out_offsets - (out_offsets // hw) * hw
    h_idx = hw_idx // W
    w_idx = hw_idx - h_idx * W
    input_offsets = (
        n_idx * stride_n
        + channel * stride_c
        + h_idx * stride_h
        + w_idx * stride_w
    )

    mask_input = tl.load(mask_input_ptr + input_offsets, mask=active, other=0.0)
    rhs = tl.load(where_rhs_ptr + input_offsets, mask=active, other=0.0)
    fill = tl.load(fill_ptr + 0)
    where_bf16 = tl.where(mask_input <= 0.0, fill, rhs)
    where_f32 = where_bf16.to(tl.float32)

    base = tl.load(centered_base_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel * mean_stride_c, mask=active, other=0.0).to(tl.float32)
    centered = _f32_sub(base, mean)

    sum_where = tl.load(sum_where_ptr + channel, mask=active, other=0.0).to(tl.float32)
    sum_centered = tl.load(sum_centered_ptr + channel, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=active, other=0.0).to(tl.float32)
    affine_rhs = tl.load(affine_rhs_ptr + channel, mask=active, other=0.0).to(tl.float32)

    mean_grad = _f32_mul(sum_where, 7.62939453125e-06)
    centered_grad = _f32_mul(
        _f32_mul(sum_centered, 7.62939453125e-06),
        _f32_mul(weight, weight),
    )
    post_scale = _f32_mul(weight, affine_rhs)

    scaled_centered = _f32_mul(centered, centered_grad)
    sub_1 = _f32_sub(where_f32, scaled_centered)
    sub_2 = _f32_sub(sub_1, mean_grad)
    out = _f32_mul(sub_2, post_scale)
    tl.store(out_ptr + out_offsets, out.to(tl.bfloat16), mask=active)


@oracle_impl(
    hardware="B200",
    point="4634de77",
    C=128,
    H=32,
    W=32,
    REDUCE_BLOCK=4096,
    C_BLOCK=1,
    FINAL_C_BLOCK=8,
    EPILOGUE_BLOCK=1024,
    reduce_warps=8,
    final_warps=4,
    epilogue_warps=4,
)
@oracle_impl(
    hardware="B200",
    point="efa66b2e",
    C=160,
    H=16,
    W=16,
    REDUCE_BLOCK=1024,
    C_BLOCK=1,
    FINAL_C_BLOCK=8,
    EPILOGUE_BLOCK=1024,
    reduce_warps=4,
    final_warps=4,
    epilogue_warps=4,
)
@oracle_impl(
    hardware="B200",
    point="912a44db",
    C=176,
    H=8,
    W=8,
    REDUCE_BLOCK=1024,
    C_BLOCK=1,
    FINAL_C_BLOCK=8,
    EPILOGUE_BLOCK=1024,
    reduce_warps=4,
    final_warps=4,
    epilogue_warps=4,
)
def oracle_forward(
    inputs,
    *,
    C: int,
    H: int,
    W: int,
    REDUCE_BLOCK: int,
    C_BLOCK: int,
    FINAL_C_BLOCK: int,
    EPILOGUE_BLOCK: int,
    reduce_warps: int,
    final_warps: int,
    epilogue_warps: int,
):
    mask_input, fill, where_rhs, centered_base, mean, weight, affine_rhs = inputs
    n = mask_input.shape[0]
    hw = H * W
    e = n * hw
    total = n * C * hw
    num_chunks = triton.cdiv(e, REDUCE_BLOCK)
    block_chunks = triton.next_power_of_2(num_chunks)

    partial_where = torch.empty((num_chunks, C), device=mask_input.device, dtype=torch.float32)
    partial_where_centered = torch.empty(
        (num_chunks, C), device=mask_input.device, dtype=torch.float32
    )
    sum_where = torch.empty_strided((C,), (1,), device=mask_input.device, dtype=torch.float32)
    grad_weight = torch.empty_strided((C,), (1,), device=mask_input.device, dtype=torch.float32)
    sum_centered = torch.empty_strided((C,), (1,), device=mask_input.device, dtype=torch.float32)
    out = torch.empty_strided(
        (n, C, H, W),
        (C * hw, hw, W, 1),
        device=mask_input.device,
        dtype=torch.bfloat16,
    )

    _partial_sums_kernel[(triton.cdiv(C, C_BLOCK), num_chunks)](
        mask_input,
        fill,
        where_rhs,
        centered_base,
        mean,
        partial_where,
        partial_where_centered,
        C,
        H,
        W,
        e,
        mask_input.stride(0),
        mask_input.stride(1),
        mask_input.stride(2),
        mask_input.stride(3),
        mean.stride(1),
        BLOCK_E=REDUCE_BLOCK,
        C_BLOCK=C_BLOCK,
        num_warps=reduce_warps,
    )
    _finalize_sums_kernel[(triton.cdiv(C, FINAL_C_BLOCK),)](
        partial_where,
        partial_where_centered,
        weight,
        sum_where,
        grad_weight,
        sum_centered,
        C,
        num_chunks,
        BLOCK_CHUNKS=block_chunks,
        C_BLOCK=FINAL_C_BLOCK,
        num_warps=final_warps,
    )
    _dense_epilogue_kernel[(triton.cdiv(total, EPILOGUE_BLOCK),)](
        mask_input,
        fill,
        where_rhs,
        centered_base,
        mean,
        weight,
        affine_rhs,
        sum_where,
        sum_centered,
        out,
        C,
        H,
        W,
        total,
        mask_input.stride(0),
        mask_input.stride(1),
        mask_input.stride(2),
        mask_input.stride(3),
        mean.stride(1),
        BLOCK=EPILOGUE_BLOCK,
        num_warps=epilogue_warps,
    )
    return sum_where, grad_weight, out, out[:, :16, :, :]
