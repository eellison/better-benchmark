"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 GPT-OSS residual-add RMSNorm scope in one Triton row kernel, storing the returned bf16 residual add, retaining that bf16-rounded tile for the fp32 mean-square reduction with eps=1e-5, applying the fp32 affine weight multiply, and emitting the final flattened bf16 view, whereas Inductor lowers the captured add/mean/rsqrt/affine/view graph through its generic fused reduction scheduler; Inductor cannot do this today because the row-reduction scheduler does not keep the returned residual-add producer live across the fixed-hidden RMSNorm reduction while preserving the required bf16 side output and epilogue store; the fix is SCHEDULER_FUSION: teach the norm-template scheduler to inline same-layout residual adds with visible side-output stores into the fused RMSNorm row plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _round_bf16_to_fp32(x):
    bits = x.to(tl.uint32, bitcast=True)
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _residual_rmsnorm_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    add_out_ptr,
    norm_out_ptr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN
    offsets = row * HIDDEN + cols

    flat = tl.load(flat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    added = _round_bf16_to_fp32(residual + flat)
    tl.store(add_out_ptr + offsets, added, mask=mask)

    square_sum = tl.sum(tl.where(mask, added * added, 0.0), axis=0)
    inv_rms = tl.rsqrt(square_sum * (1.0 / HIDDEN) + 1.0e-5)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = weight * (added * inv_rms)
    tl.store(norm_out_ptr + offsets, out, mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


# 533b2091: (T([1000,2880], bf16), T([1,1000,2880], bf16), T([2880], bf16), S([1,1000,2880]), S([-1,2880]))
@oracle_impl(hardware="B200", point="533b2091", BLOCK_H=4096, num_warps=4)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, shape0, shape1 = inputs
    add_shape = tuple(int(dim) for dim in shape0)

    add_out = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_base = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _residual_rmsnorm_kernel[(int(arg0_1.shape[0]),)](
        arg0_1,
        arg1_1,
        arg2_1,
        add_out,
        norm_base,
        HIDDEN=int(arg0_1.shape[1]),
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=2,
    )
    return add_out, norm_base.view(tuple(int(dim) for dim in shape1))
