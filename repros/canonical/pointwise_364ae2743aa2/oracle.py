"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 GhostNet/MobileNetV3 hard-sigmoid broadcast multiply with shape-specialized Triton kernels that write the returned channels-last bf16 output layout directly, using a storage-linear path for the 960x7x7 point and channel/spatial tiling elsewhere to reuse singleton-spatial gate values, whereas Inductor lowers the same cast/add/clamp/div/mul scope as a generic storage-linear pointwise kernel that reloads the broadcast gate for every activation element; Inductor cannot do this today because pointwise scheduling has no B200-tuned singleton-spatial hard-sigmoid broadcast template that can choose between flat storage order and channel/spatial reuse while preserving the fused fp32 arithmetic and final bf16 store; the fix is NEW_PATTERN: add a guarded hard-sigmoid-broadcast-mul pointwise template or scheduler rule for channels-last CNN gates when it beats the generic flat schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _flat_hardsigmoid_broadcast_mul_kernel(
    gate_ptr,
    x_ptr,
    out_ptr,
    gate_s0: tl.constexpr,
    gate_s1: tl.constexpr,
    channels: tl.constexpr,
    hw: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c_offsets = offsets % channels
    n_offsets = offsets // (channels * hw)

    x = tl.load(x_ptr + offsets).to(tl.float32)
    gate = tl.load(
        gate_ptr + n_offsets * gate_s0 + c_offsets * gate_s1,
        eviction_policy="evict_last",
    ).to(tl.float32)
    shifted = gate + 3.0
    clamped = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
    scale = clamped * 0.16666666666666666
    tl.store(out_ptr + offsets, x * scale)


@triton.jit
def _hardsigmoid_broadcast_mul_kernel(
    gate_ptr,
    x_ptr,
    out_ptr,
    gate_s0: tl.constexpr,
    gate_s1: tl.constexpr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    x_s3: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    out_s3: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    hw_tiles: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    nhw_tile = tl.program_id(1)
    n_idx = nhw_tile // hw_tiles
    hw_base = (nhw_tile - n_idx * hw_tiles) * BLOCK_HW
    hw_offsets = hw_base + tl.arange(0, BLOCK_HW)

    hw = height * width
    h_offsets = hw_offsets // width
    w_offsets = hw_offsets - h_offsets * width

    c_mask = c_offsets < channels
    hw_mask = hw_offsets < hw

    gate = tl.load(
        gate_ptr + n_idx * gate_s0 + c_offsets * gate_s1,
        mask=c_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    shifted = gate + 3.0
    clamped = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
    scale = clamped * 0.16666666666666666

    x_offsets = (
        n_idx * x_s0
        + h_offsets[:, None] * x_s2
        + w_offsets[:, None] * x_s3
        + c_offsets[None, :] * x_s1
    )
    out_offsets = (
        n_idx * out_s0
        + h_offsets[:, None] * out_s2
        + w_offsets[:, None] * out_s3
        + c_offsets[None, :] * out_s1
    )
    mask = hw_mask[:, None] & c_mask[None, :]

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + out_offsets, x * scale[None, :], mask=mask)


# c72422f8: (T([512,960,1,1], bf16), T([512,960,7,7], bf16, stride=(47040,1,6720,960)))
@oracle_impl(hardware="B200", point="c72422f8", BLOCK_C=128, BLOCK_HW=16, USE_FLAT=True, num_warps=4)
# 3fb6168e: (T([512,672,1,1], bf16), T([512,672,7,7], bf16, stride=(32928,1,4704,672)))
@oracle_impl(hardware="B200", point="3fb6168e", BLOCK_C=128, BLOCK_HW=16, USE_FLAT=False, num_warps=4)
# 33f8ec7a: (T([512,672,1,1], bf16), T([512,672,14,14], bf16, stride=(131712,1,9408,672)))
@oracle_impl(hardware="B200", point="33f8ec7a", BLOCK_C=128, BLOCK_HW=16, USE_FLAT=False, num_warps=4)
# c6ad2c20: (T([512,480,1,1], bf16), T([512,480,14,14], bf16, stride=(94080,1,6720,480)))
@oracle_impl(hardware="B200", point="c6ad2c20", BLOCK_C=128, BLOCK_HW=16, USE_FLAT=False, num_warps=4)
# 2e45fbca: (T([512,120,1,1], bf16), T([512,120,28,28], bf16, stride=(94080,1,3360,120)))
@oracle_impl(hardware="B200", point="2e45fbca", BLOCK_C=128, BLOCK_HW=16, USE_FLAT=False, num_warps=4)
# c8968885: (T([512,72,1,1], bf16), T([512,72,28,28], bf16, stride=(56448,1,2016,72)))
@oracle_impl(hardware="B200", point="c8968885", BLOCK_C=128, BLOCK_HW=16, USE_FLAT=False, num_warps=4)
def oracle_forward(inputs, *, BLOCK_C, BLOCK_HW, USE_FLAT, num_warps):
    gate, x = inputs
    output = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )

    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    if USE_FLAT:
        block = 1024
        grid = (x.numel() // block,)
        _flat_hardsigmoid_broadcast_mul_kernel[grid](
            gate,
            x,
            output,
            gate_s0=int(gate.stride(0)),
            gate_s1=int(gate.stride(1)),
            channels=channels,
            hw=height * width,
            BLOCK=block,
            num_warps=num_warps,
        )
        return output

    hw_tiles = triton.cdiv(height * width, BLOCK_HW)
    grid = (triton.cdiv(channels, BLOCK_C), batch * hw_tiles)
    _hardsigmoid_broadcast_mul_kernel[grid](
        gate,
        x,
        output,
        gate_s0=int(gate.stride(0)),
        gate_s1=int(gate.stride(1)),
        x_s0=int(x.stride(0)),
        x_s1=int(x.stride(1)),
        x_s2=int(x.stride(2)),
        x_s3=int(x.stride(3)),
        out_s0=int(output.stride(0)),
        out_s1=int(output.stride(1)),
        out_s2=int(output.stride(2)),
        out_s3=int(output.stride(3)),
        channels=channels,
        height=height,
        width=width,
        hw_tiles=hw_tiles,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
    )
    return output
