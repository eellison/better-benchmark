"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet bf16 avg-pool-backward SiLU-gradient fanout, including the returned channels-last bf16 tensors, the returned bf16 gate derivative input, the bf16-rounded spatial reduction epilogue, and the final returned f32 channel vector, whereas Inductor schedules the avg-pool backward, pointwise fanout, spatial sum, gate epilogue, and channel sum as separate generic regions around materialized intermediates; Inductor cannot do this today because its scheduler does not form a full-scope multi-output reduction template that sinks the deterministic pool-backward scatter into the fused channels-last producer while preserving bf16 casts and sibling returned outputs; the fix is SCHEDULER_FUSION: add a guarded NFNet pool-backward SiLU fanout schedule that emits all required stores and dependent reductions from one fused loop nest plus a small finalizer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _zero_kernel(
    out_ptr,
    total_elems: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offsets, tl.zeros((BLOCK,), dtype=tl.float32), mask=offsets < total_elems)


@triton.jit
def _pool_silu_spatial_split_kernel(
    pool_grad_ptr,
    residual_ptr,
    activation_ptr,
    reduce_rhs_ptr,
    silu_out_ptr,
    scaled_out_ptr,
    partial_ptr,
    channels: tl.constexpr,
    width: tl.constexpr,
    pool_width: tl.constexpr,
    hw_size: tl.constexpr,
    pool_s0: tl.constexpr,
    pool_s1: tl.constexpr,
    pool_s2: tl.constexpr,
    pool_s3: tl.constexpr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    x_s3: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c_block = tl.program_id(1)
    hw_block = tl.program_id(2)

    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    hw = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)[:, None]
    c_b = c[None, :]

    c_mask = c < channels
    hw_mask = hw < hw_size
    mask = hw_mask & c_mask[None, :]

    h = hw // width
    w = hw - h * width
    pool_h = h // 2
    pool_w = w // 2

    full_offsets = n * x_s0 + c_b * x_s1 + h * x_s2 + w * x_s3
    pool_offsets = n * pool_s0 + c_b * pool_s1 + pool_h * pool_s2 + pool_w * pool_s3

    pool_grad = tl.load(pool_grad_ptr + pool_offsets, mask=mask, other=0.0).to(tl.float32)
    pool = pool_grad * 0.25
    pool_bf16 = pool.to(tl.bfloat16)
    residual = tl.load(residual_ptr + full_offsets, mask=mask, other=0.0).to(tl.float32)
    added = residual + pool
    scaled = added * 0.9622504486493761
    added_bf16 = (residual + pool_bf16.to(tl.float32)).to(tl.bfloat16)
    scaled_bf16 = (added_bf16.to(tl.float32) * 0.9622504486493761).to(tl.bfloat16)

    activation = tl.load(activation_ptr + full_offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid = tl.sigmoid(activation)
    silu_factor = sigmoid * (activation * (1.0 - sigmoid) + 1.0)
    silu_store_bf16 = (scaled * silu_factor).to(tl.bfloat16)
    silu_bf16 = (scaled_bf16.to(tl.float32) * silu_factor).to(tl.bfloat16)

    mul4_bf16 = (silu_bf16.to(tl.float32) * 0.2).to(tl.bfloat16)
    scaled_bf16_out = (mul4_bf16.to(tl.float32) * 2.0).to(tl.bfloat16)
    reduce_rhs = tl.load(reduce_rhs_ptr + full_offsets, mask=mask, other=0.0).to(tl.float32)
    reduce_val_bf16 = (scaled_bf16_out.to(tl.float32) * reduce_rhs).to(tl.bfloat16)

    tl.store(silu_out_ptr + full_offsets, silu_store_bf16, mask=mask)
    tl.store(scaled_out_ptr + full_offsets, scaled_bf16_out, mask=mask)

    spatial_sum = tl.sum(tl.where(mask, reduce_val_bf16.to(tl.float32), 0.0), axis=0)
    tl.atomic_add(
        partial_ptr + n * channels + c,
        spatial_sum,
        sem="relaxed",
        mask=c_mask,
    )


@triton.jit
def _gate_row_channel_finalizer_kernel(
    partial_ptr,
    gate_ptr,
    gate_out_ptr,
    row_out_ptr,
    final_out_ptr,
    channels: tl.constexpr,
    batch: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    n = tl.arange(0, BLOCK_N)[:, None]
    c_b = c[None, :]

    c_mask = c < channels
    n_mask = n < batch
    mask = n_mask & c_mask[None, :]
    offsets = n * channels + c_b

    spatial_sum = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate_sig = tl.sigmoid(gate).to(tl.bfloat16)
    gate_sig_f32 = gate_sig.to(tl.float32)
    gate_deriv = gate_sig_f32 * (1.0 - gate_sig_f32)

    sum_bf16_f32 = spatial_sum.to(tl.bfloat16).to(tl.float32)
    row_bf16 = (sum_bf16_f32 * gate_deriv).to(tl.bfloat16)

    tl.store(gate_out_ptr + offsets, gate_sig, mask=mask)
    tl.store(row_out_ptr + offsets, row_bf16, mask=mask)

    total = tl.sum(tl.where(mask, row_bf16.to(tl.float32), 0.0), axis=0)
    tl.store(final_out_ptr + c, total.to(tl.bfloat16).to(tl.float32), mask=c_mask)


def _oracle_forward_impl(
    inputs,
    *,
    BLOCK_HW: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    del arg1_1

    batch = int(arg2_1.shape[0])
    channels = int(arg2_1.shape[1])
    height = int(arg2_1.shape[2])
    width = int(arg2_1.shape[3])
    hw_size = height * width

    silu_out = torch.empty_strided(
        tuple(arg2_1.shape),
        tuple(arg2_1.stride()),
        device=arg2_1.device,
        dtype=torch.bfloat16,
    )
    scaled_out = torch.empty_strided(
        tuple(arg2_1.shape),
        tuple(arg2_1.stride()),
        device=arg2_1.device,
        dtype=torch.bfloat16,
    )
    gate_out = torch.empty_strided(
        tuple(arg5_1.shape),
        tuple(arg5_1.stride()),
        device=arg5_1.device,
        dtype=torch.bfloat16,
    )
    row_out = torch.empty_strided(
        tuple(arg5_1.shape),
        tuple(arg5_1.stride()),
        device=arg5_1.device,
        dtype=torch.bfloat16,
    )
    final_out = torch.empty_strided(
        (channels,),
        (1,),
        device=arg2_1.device,
        dtype=torch.float32,
    )
    partial = torch.empty_strided(
        tuple(arg5_1.shape),
        tuple(arg5_1.stride()),
        device=arg5_1.device,
        dtype=torch.float32,
    )

    _zero_kernel[(triton.cdiv(batch * channels, 1024),)](
        partial,
        total_elems=batch * channels,
        BLOCK=1024,
        num_warps=4,
        num_stages=3,
    )
    _pool_silu_spatial_split_kernel[
        (
            batch,
            triton.cdiv(channels, BLOCK_C),
            triton.cdiv(hw_size, BLOCK_HW),
        )
    ](
        arg0_1,
        arg2_1,
        arg3_1,
        arg4_1,
        silu_out,
        scaled_out,
        partial,
        channels=channels,
        width=width,
        pool_width=int(arg0_1.shape[3]),
        hw_size=hw_size,
        pool_s0=arg0_1.stride(0),
        pool_s1=arg0_1.stride(1),
        pool_s2=arg0_1.stride(2),
        pool_s3=arg0_1.stride(3),
        x_s0=arg2_1.stride(0),
        x_s1=arg2_1.stride(1),
        x_s2=arg2_1.stride(2),
        x_s3=arg2_1.stride(3),
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=4,
    )
    _gate_row_channel_finalizer_kernel[
        (triton.cdiv(channels, FINAL_BLOCK_C),)
    ](
        partial,
        arg5_1,
        gate_out,
        row_out,
        final_out,
        channels=channels,
        batch=batch,
        BLOCK_N=128,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=4,
        num_stages=3,
    )

    return silu_out, scaled_out, gate_out, row_out, final_out


# 2570be15: NFNet-L0 train, bf16 [128,512,28,28] channels-last fanout.
@oracle_impl(hardware="B200", point="2570be15", BLOCK_HW=128, BLOCK_C=64, FINAL_BLOCK_C=16, num_warps=8)
# 1f32ce1f: NFNet-L0 train, bf16 [128,1536,14,14] channels-last fanout.
@oracle_impl(hardware="B200", point="1f32ce1f", BLOCK_HW=128, BLOCK_C=128, FINAL_BLOCK_C=16, num_warps=8)
def oracle_forward(
    inputs,
    *,
    BLOCK_HW: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    return _oracle_forward_impl(
        inputs,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        FINAL_BLOCK_C=FINAL_BLOCK_C,
        num_warps=num_warps,
    )
