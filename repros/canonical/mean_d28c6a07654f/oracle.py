"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 residual-add RMSNorm alias scope in one Triton row kernel, storing the returned bf16 residual add, retaining that bf16-rounded tile for the fp32 mean-square and eps=1e-6 rsqrt normalization, and emitting both final alias views from one output buffer, whereas Inductor lowers the same graph through a generic reduction schedule that reloads or recomputes the residual-add producer for the normalization epilogue; Inductor cannot do this today because its row-reduction scheduler does not keep the full hidden tile's producer values live across the reduction while also preserving a returned side output and duplicate view aliases; the fix is SCHEDULER_FUSION: add a guarded residual-RMSNorm row schedule that sinks returned side-output stores and alias-view epilogues into the fused normalization kernel."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _residual_rmsnorm_kernel(
    mm_ptr,
    residual_ptr,
    weight_ptr,
    add_out_ptr,
    norm_out_ptr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    offsets = row * HIDDEN + cols

    mm = tl.load(mm_ptr + offsets).to(tl.float32)
    residual = tl.load(residual_ptr + offsets).to(tl.float32)
    add_bf16 = (mm + residual).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, add_bf16)

    x = add_bf16.to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(sum_sq * (1.0 / HIDDEN) + 1.0e-6)
    norm_bf16 = (x * inv_rms).to(tl.bfloat16)

    weight = tl.load(weight_ptr + cols).to(tl.float32)
    out = (weight * norm_bf16.to(tl.float32)).to(tl.bfloat16)
    tl.store(norm_out_ptr + offsets, out)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


# da8b94aa: (T([1000, 1024], bf16), T([1, 1000, 1024], bf16), T([1024], bf16), ...)
# 111af936: (T([1000, 4096], bf16), T([1, 1000, 4096], bf16), T([4096], bf16), ...)
@oracle_impl(hardware="B200", point="da8b94aa", BLOCK_H=1024, num_warps=4)
@oracle_impl(hardware="B200", point="111af936", BLOCK_H=4096, num_warps=8)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int):
    mm, residual, weight, add_shape, out_shape0, out_shape1 = inputs
    rows = int(mm.shape[0])
    hidden = int(mm.shape[1])

    add_shape = tuple(int(dim) for dim in add_shape)
    out_base = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=mm.device,
        dtype=torch.bfloat16,
    )
    norm_base = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=mm.device,
        dtype=torch.bfloat16,
    )

    _residual_rmsnorm_kernel[(rows,)](
        mm,
        residual,
        weight,
        out_base,
        norm_base,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=2,
    )
    return (
        out_base,
        norm_base.view(tuple(int(dim) for dim in out_shape0)),
        norm_base.view(tuple(int(dim) for dim in out_shape1)),
    )
