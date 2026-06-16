"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete SqueezeNet bf16 split-channel mask/where/sum scope by materializing both returned channels-last `where` tensors while co-reducing the lower and upper channel halves from the shared bool-scaled producer, whereas Inductor lowers the bool-to-bf16 scale, channel slices, sibling masked materializations, bf16 reductions, and fp32 vector casts as generic regions over the same N/H/W iteration space; Inductor cannot do this today because its scheduler/codegen does not form one full-scope multi-output plan that shares the compatible channels-last producers and preserves the final bf16 sum rounding boundary before the returned fp32 vectors; the fix is SCHEDULER_FUSION: teach reduction scheduling to fuse compatible sibling masked reductions with their dense materialized outputs into one multi-accumulator channels-last template."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 32
FULL_C = 512
HALF_C = 256
H = 13
W = 13
HW = H * W
K_TOTAL = N * HW
_USE_INDUCTOR_NUMERICS = False


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


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
def _scaled_bf16_value(bool_value, x_value):
    scale = _f32_mul(bool_value.to(tl.float32), 2.0).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    return _f32_mul(x_value.to(tl.float32), scale.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )


@triton.jit
def _dual_scaled_where_partial_kernel(
    scale_mask_ptr,
    x_ptr,
    upper_mask_ptr,
    fill_ptr,
    lower_mask_ptr,
    upper_out_ptr,
    lower_out_ptr,
    partial_ptr,
    INPUT_STRIDE_N: tl.constexpr,
    INPUT_STRIDE_H: tl.constexpr,
    INPUT_STRIDE_W: tl.constexpr,
    MASK_STRIDE_N: tl.constexpr,
    MASK_STRIDE_H: tl.constexpr,
    MASK_STRIDE_W: tl.constexpr,
    C_N: tl.constexpr,
    W_N: tl.constexpr,
    HW_N: tl.constexpr,
    K_TOTAL_N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    active_k = k_offsets < K_TOTAL_N
    active_c = c_offsets < C_N
    active = active_k[:, None] & active_c[None, :]

    n = k_offsets // HW_N
    spatial = k_offsets - n * HW_N
    h = spatial // W_N
    w = spatial - h * W_N

    lower_offsets = (
        n[:, None] * INPUT_STRIDE_N
        + h[:, None] * INPUT_STRIDE_H
        + w[:, None] * INPUT_STRIDE_W
        + c_offsets[None, :]
    )
    upper_offsets = lower_offsets + C_N
    out_offsets = (
        n[:, None] * MASK_STRIDE_N
        + h[:, None] * MASK_STRIDE_H
        + w[:, None] * MASK_STRIDE_W
        + c_offsets[None, :]
    )

    fill = tl.load(fill_ptr)
    lower_scale_mask = tl.load(scale_mask_ptr + lower_offsets, mask=active, other=0)
    upper_scale_mask = tl.load(scale_mask_ptr + upper_offsets, mask=active, other=0)
    lower_x = tl.load(x_ptr + lower_offsets, mask=active, other=0.0)
    upper_x = tl.load(x_ptr + upper_offsets, mask=active, other=0.0)

    lower_scaled = _scaled_bf16_value(lower_scale_mask, lower_x)
    upper_scaled = _scaled_bf16_value(upper_scale_mask, upper_x)
    upper_mask = tl.load(upper_mask_ptr + out_offsets, mask=active, other=0)
    lower_mask = tl.load(lower_mask_ptr + out_offsets, mask=active, other=0)
    upper_where = tl.where(upper_mask, fill, upper_scaled)
    lower_where = tl.where(lower_mask, fill, lower_scaled)

    tl.store(upper_out_ptr + out_offsets, upper_where, mask=active)
    tl.store(lower_out_ptr + out_offsets, lower_where, mask=active)

    upper_sum = tl.sum(tl.where(active, upper_where.to(tl.float32), 0.0), axis=0)
    lower_sum = tl.sum(tl.where(active, lower_where.to(tl.float32), 0.0), axis=0)
    partial_offsets = c_offsets * NUM_K_BLOCKS + tl.program_id(1)
    tl.store(partial_ptr + partial_offsets, upper_sum, mask=active_c)
    tl.store(
        partial_ptr + C_N * NUM_K_BLOCKS + partial_offsets,
        lower_sum,
        mask=active_c,
    )


@triton.jit
def _dual_scaled_where_final_kernel(
    partial_ptr,
    upper_sum_out_ptr,
    lower_sum_out_ptr,
    C_N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    USE_INDUCTOR_NUMERICS: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    block_offsets = tl.arange(0, BLOCK_BLOCKS)
    active = (block_offsets[:, None] < NUM_K_BLOCKS) & (
        c_offsets[None, :] < C_N
    )
    partial_offsets = c_offsets[None, :] * NUM_K_BLOCKS + block_offsets[:, None]

    upper_parts = tl.load(partial_ptr + partial_offsets, mask=active, other=0.0)
    lower_parts = tl.load(
        partial_ptr + C_N * NUM_K_BLOCKS + partial_offsets,
        mask=active,
        other=0.0,
    )
    upper_sum = tl.sum(upper_parts.to(tl.float32), axis=0)
    lower_sum = tl.sum(lower_parts.to(tl.float32), axis=0)
    if not USE_INDUCTOR_NUMERICS:
        upper_sum = upper_sum.to(tl.bfloat16, fp_downcast_rounding="rtne").to(
            tl.float32
        )
        lower_sum = lower_sum.to(tl.bfloat16, fp_downcast_rounding="rtne").to(
            tl.float32
        )

    out_mask = c_offsets < C_N
    tl.store(upper_sum_out_ptr + c_offsets, upper_sum, mask=out_mask)
    tl.store(lower_sum_out_ptr + c_offsets, lower_sum, mask=out_mask)


# 12a31f97: torchbench squeezenet1_1 train, bf16/bool channels-last C split.
@oracle_impl(
    hardware="B200",
    point="12a31f97",
    BLOCK_K=256,
    BLOCK_C=16,
    FINAL_BLOCK_C=16,
    num_warps=8,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    global _USE_INDUCTOR_NUMERICS
    scale_mask, x, upper_mask, fill, lower_mask = inputs
    use_inductor_numerics = _USE_INDUCTOR_NUMERICS
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _USE_INDUCTOR_NUMERICS = True
        use_inductor_numerics = True
    num_k_blocks = triton.cdiv(K_TOTAL, BLOCK_K)
    block_blocks = _next_power_of_2(num_k_blocks)

    out_stride = (HALF_C * HW, 1, W * HALF_C, HALF_C)
    upper_out = torch.empty_strided(
        (N, HALF_C, H, W),
        out_stride,
        device=x.device,
        dtype=torch.bfloat16,
    )
    lower_out = torch.empty_strided(
        (N, HALF_C, H, W),
        out_stride,
        device=x.device,
        dtype=torch.bfloat16,
    )
    upper_sum = torch.empty((HALF_C,), device=x.device, dtype=torch.float32)
    lower_sum = torch.empty((HALF_C,), device=x.device, dtype=torch.float32)
    partial = torch.empty(
        (2, HALF_C, num_k_blocks),
        device=x.device,
        dtype=torch.float32,
    )

    _dual_scaled_where_partial_kernel[(triton.cdiv(HALF_C, BLOCK_C), num_k_blocks)](
        scale_mask,
        x,
        upper_mask,
        fill,
        lower_mask,
        upper_out,
        lower_out,
        partial,
        INPUT_STRIDE_N=x.stride(0),
        INPUT_STRIDE_H=x.stride(2),
        INPUT_STRIDE_W=x.stride(3),
        MASK_STRIDE_N=upper_mask.stride(0),
        MASK_STRIDE_H=upper_mask.stride(2),
        MASK_STRIDE_W=upper_mask.stride(3),
        C_N=HALF_C,
        W_N=W,
        HW_N=HW,
        K_TOTAL_N=K_TOTAL,
        NUM_K_BLOCKS=num_k_blocks,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    _dual_scaled_where_final_kernel[(triton.cdiv(HALF_C, FINAL_BLOCK_C),)](
        partial,
        upper_sum,
        lower_sum,
        C_N=HALF_C,
        NUM_K_BLOCKS=num_k_blocks,
        BLOCK_BLOCKS=block_blocks,
        BLOCK_C=FINAL_BLOCK_C,
        USE_INDUCTOR_NUMERICS=use_inductor_numerics,
        num_warps=4,
        num_stages=3,
    )
    return upper_out, upper_sum, lower_out, lower_sum
