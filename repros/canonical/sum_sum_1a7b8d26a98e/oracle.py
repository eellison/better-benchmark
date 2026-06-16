"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet bf16 SiLU-gradient fanout, including the explicit bf16 round before the residual add, both returned full-layout bf16 tensors, the returned bf16 sigmoid gate, the bf16 row reduction epilogue, and the final returned f32 channel vector in Triton kernels over the captured channels-last activation strides, whereas Inductor currently schedules the decomposed pointwise fanout, spatial sum, sigmoid-derivative epilogue, and channel sum as generic regions around materialized intermediates; Inductor cannot do this today because its scheduler does not form one full-scope multi-output reduction template that preserves bf16 rounding boundaries while fusing sibling materialized outputs and dependent reductions across non-contiguous dense strides; the fix is SCHEDULER_FUSION: add a guarded NFNet SiLU-gradient fanout template that emits all returned stores and the dependent reductions from one fused loop nest plus a small finalizer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _nfnet_silu_spatial_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    add_out_ptr,
    mul5_out_ptr,
    gate_out_ptr,
    row_out_ptr,
    total_rows: tl.constexpr,
    channels: tl.constexpr,
    width: tl.constexpr,
    hw_size: tl.constexpr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    x_s3: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    c_block = tl.program_id(1)
    n = row
    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    hw = tl.arange(0, BLOCK_HW)[:, None]
    c_b = c[None, :]

    c_mask = c < channels
    hw_mask = hw < hw_size
    mask = hw_mask & c_mask[None, :]

    h = hw // width
    w = hw - h * width
    offsets = n * x_s0 + c_b * x_s1 + h * x_s2 + w * x_s3

    arg0 = tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    arg1 = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    arg2 = tl.load(arg2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    arg3 = tl.load(arg3_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    scaled = (arg0 * 0.9622504486493761).to(tl.bfloat16).to(tl.float32)
    sig = tl.sigmoid(arg1)
    silu_grad = scaled * sig * (arg1 * (1.0 - sig) + 1.0)
    silu_grad_bf16 = silu_grad.to(tl.bfloat16)

    add_bf16 = (arg2 + silu_grad_bf16.to(tl.float32)).to(tl.bfloat16)
    add_f32 = add_bf16.to(tl.float32)
    mul4_bf16 = (add_f32 * 0.2).to(tl.bfloat16)
    mul5_bf16 = (mul4_bf16.to(tl.float32) * 2.0).to(tl.bfloat16)
    mul6_bf16 = (mul5_bf16.to(tl.float32) * arg3).to(tl.bfloat16)

    tl.store(add_out_ptr + offsets, add_bf16, mask=mask)
    tl.store(mul5_out_ptr + offsets, mul5_bf16, mask=mask)

    spatial_sum = tl.sum(
        tl.where(mask, mul6_bf16.to(tl.float32), 0.0),
        axis=0,
    )

    gate_offsets = n * channels + c
    gate = tl.load(arg4_ptr + gate_offsets, mask=c_mask, other=0.0).to(tl.float32)
    gate_sig = tl.sigmoid(gate).to(tl.bfloat16)
    gate_sig_f32 = gate_sig.to(tl.float32)
    gate_deriv = gate_sig_f32 * (1.0 - gate_sig_f32)

    sum_bf16_f32 = spatial_sum.to(tl.bfloat16).to(tl.float32)
    row_bf16 = (sum_bf16_f32 * gate_deriv).to(tl.bfloat16)

    tl.store(gate_out_ptr + gate_offsets, gate_sig, mask=c_mask)
    tl.store(row_out_ptr + gate_offsets, row_bf16, mask=c_mask)


@triton.jit
def _zero_kernel(
    out_ptr,
    total_elems: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offsets, tl.zeros((BLOCK,), dtype=tl.float32), mask=offsets < total_elems)


@triton.jit
def _nfnet_silu_spatial_split_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    add_out_ptr,
    mul5_out_ptr,
    partial_ptr,
    channels: tl.constexpr,
    width: tl.constexpr,
    hw_size: tl.constexpr,
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
    offsets = n * x_s0 + c_b * x_s1 + h * x_s2 + w * x_s3

    arg0 = tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    arg1 = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    arg2 = tl.load(arg2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    arg3 = tl.load(arg3_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    scaled = (arg0 * 0.9622504486493761).to(tl.bfloat16).to(tl.float32)
    sig = tl.sigmoid(arg1)
    silu_grad = scaled * sig * (arg1 * (1.0 - sig) + 1.0)
    silu_grad_bf16 = silu_grad.to(tl.bfloat16)

    add_bf16 = (arg2 + silu_grad_bf16.to(tl.float32)).to(tl.bfloat16)
    add_f32 = add_bf16.to(tl.float32)
    mul4_bf16 = (add_f32 * 0.2).to(tl.bfloat16)
    mul5_bf16 = (mul4_bf16.to(tl.float32) * 2.0).to(tl.bfloat16)
    mul6_bf16 = (mul5_bf16.to(tl.float32) * arg3).to(tl.bfloat16)

    tl.store(add_out_ptr + offsets, add_bf16, mask=mask)
    tl.store(mul5_out_ptr + offsets, mul5_bf16, mask=mask)

    spatial_sum = tl.sum(
        tl.where(mask, mul6_bf16.to(tl.float32), 0.0),
        axis=0,
    )
    tl.atomic_add(
        partial_ptr + n * channels + c,
        spatial_sum,
        sem="relaxed",
        mask=c_mask,
    )


@triton.jit
def _gate_row_channel_finalizer_kernel(
    partial_ptr,
    arg4_ptr,
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
    gate = tl.load(arg4_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate_sig = tl.sigmoid(gate).to(tl.bfloat16)
    gate_sig_f32 = gate_sig.to(tl.float32)
    gate_deriv = gate_sig_f32 * (1.0 - gate_sig_f32)

    sum_bf16_f32 = spatial_sum.to(tl.bfloat16).to(tl.float32)
    row_bf16 = (sum_bf16_f32 * gate_deriv).to(tl.bfloat16)

    tl.store(gate_out_ptr + offsets, gate_sig, mask=mask)
    tl.store(row_out_ptr + offsets, row_bf16, mask=mask)

    total = tl.sum(tl.where(mask, row_bf16.to(tl.float32), 0.0), axis=0)
    tl.store(final_out_ptr + c, total.to(tl.bfloat16).to(tl.float32), mask=c_mask)


@triton.jit
def _channel_finalizer_kernel(
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
    c_mask = c < channels
    n_mask = n < batch
    vals = tl.load(
        row_out_ptr + n * channels + c[None, :],
        mask=n_mask & c_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    total = tl.sum(vals, axis=0)
    final_bf16 = total.to(tl.bfloat16)
    tl.store(final_out_ptr + c, final_bf16.to(tl.float32), mask=c_mask)


def _oracle_forward_impl(
    inputs,
    *,
    BLOCK_HW: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
    split_hw: bool,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = inputs
    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw_size = height * width
    total_rows = batch

    add_out = torch.empty_strided(
        tuple(arg2_1.shape),
        tuple(arg2_1.stride()),
        device=arg2_1.device,
        dtype=torch.bfloat16,
    )
    mul5_out = torch.empty_strided(
        tuple(arg2_1.shape),
        tuple(arg2_1.stride()),
        device=arg2_1.device,
        dtype=torch.bfloat16,
    )
    gate_out = torch.empty_strided(
        tuple(arg4_1.shape),
        tuple(arg4_1.stride()),
        device=arg4_1.device,
        dtype=torch.bfloat16,
    )
    row_out = torch.empty_strided(
        tuple(arg4_1.shape),
        tuple(arg4_1.stride()),
        device=arg4_1.device,
        dtype=torch.bfloat16,
    )
    final_out = torch.empty_strided(
        (channels,),
        (1,),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    if split_hw:
        partial = torch.empty_strided(
            tuple(arg4_1.shape),
            tuple(arg4_1.stride()),
            device=arg4_1.device,
            dtype=torch.float32,
        )
        _zero_kernel[(triton.cdiv(batch * channels, 1024),)](
            partial,
            total_elems=batch * channels,
            BLOCK=1024,
            num_warps=4,
            num_stages=3,
        )
        _nfnet_silu_spatial_split_kernel[
            (
                total_rows,
                triton.cdiv(channels, BLOCK_C),
                triton.cdiv(hw_size, BLOCK_HW),
            )
        ](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            add_out,
            mul5_out,
            partial,
            channels=channels,
            width=width,
            hw_size=hw_size,
            x_s0=arg0_1.stride(0),
            x_s1=arg0_1.stride(1),
            x_s2=arg0_1.stride(2),
            x_s3=arg0_1.stride(3),
            BLOCK_HW=BLOCK_HW,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
            num_stages=4,
        )
        _gate_row_channel_finalizer_kernel[
            (triton.cdiv(channels, FINAL_BLOCK_C),)
        ](
            partial,
            arg4_1,
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
    else:
        _nfnet_silu_spatial_kernel[
            (total_rows, triton.cdiv(channels, BLOCK_C))
        ](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            add_out,
            mul5_out,
            gate_out,
            row_out,
            total_rows=total_rows,
            channels=channels,
            width=width,
            hw_size=hw_size,
            x_s0=arg0_1.stride(0),
            x_s1=arg0_1.stride(1),
            x_s2=arg0_1.stride(2),
            x_s3=arg0_1.stride(3),
            BLOCK_HW=BLOCK_HW,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
            num_stages=4,
        )

        _channel_finalizer_kernel[
            (triton.cdiv(channels, FINAL_BLOCK_C),)
        ](
            row_out,
            final_out,
            channels=channels,
            batch=batch,
            BLOCK_N=128,
            BLOCK_C=FINAL_BLOCK_C,
            num_warps=4,
            num_stages=3,
        )

    return add_out, mul5_out, gate_out, row_out, final_out


# 3e2b5914: (T([128,1536,14,14], bf16, stride=(301056,1,21504,1536)), ...)
@oracle_impl(hardware="B200", point="3e2b5914", BLOCK_HW=128, BLOCK_C=128, FINAL_BLOCK_C=16, num_warps=8, split_hw=True)
# 31d48416: (T([128,1536,7,7], bf16, stride=(75264,1,10752,1536)), ...)
@oracle_impl(hardware="B200", point="31d48416", BLOCK_HW=64, BLOCK_C=64, FINAL_BLOCK_C=16, num_warps=8, split_hw=False)
def oracle_forward(
    inputs,
    *,
    BLOCK_HW: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
    split_hw: bool,
):
    return _oracle_forward_impl(
        inputs,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        FINAL_BLOCK_C=FINAL_BLOCK_C,
        num_warps=num_warps,
        split_hw=split_hw,
    )
