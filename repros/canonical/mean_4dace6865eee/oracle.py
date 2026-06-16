"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 T5 residual-add RMSNorm alias scope in one Triton row kernel, retaining the bf16-rounded residual add for the fp32 mean-square reduction, applying eps=1e-6 rsqrt normalization with the required bf16 normalization and affine rounding boundaries, and returning the base normalized tensor plus twelve flattened alias views from one storage, whereas Inductor lowers the captured view/add/mean/rsqrt/affine/repeated-view graph through its generic fused reduction scheduler; Inductor cannot do this today because the normalization scheduler does not keep the same-layout residual-add producer live across the fixed-hidden row reduction while preserving bf16 cast boundaries and many alias-only sibling returns; the fix is SCHEDULER_FUSION: teach the norm-template scheduler to inline visible same-layout residual adds into the RMSNorm row plan and emit repeated view returns from the single normalized output storage."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 2}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 4}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 8}, num_warps=8, num_stages=2),
    ],
    key=["ROWS", "HIDDEN"],
)
@triton.jit
def _residual_rmsnorm_bf16_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    norm_out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_H)
    offsets = rows[:, None] * HIDDEN + cols[None, :]

    flat = tl.load(flat_ptr + offsets).to(tl.float32)
    residual = tl.load(residual_ptr + offsets).to(tl.float32)
    added_bf16 = (residual + flat).to(tl.bfloat16)

    x = added_bf16.to(tl.float32)
    square_sum = tl.sum(x * x, axis=1)
    inv_rms = tl.rsqrt(square_sum / HIDDEN + 1.0e-6)
    normalized_bf16 = (x * inv_rms[:, None]).to(tl.bfloat16)

    weight = tl.load(weight_ptr + cols).to(tl.float32)
    out = (weight[None, :] * normalized_bf16.to(tl.float32)).to(tl.bfloat16)
    tl.store(norm_out_ptr + offsets, out)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


# 40057a60: (T([8192,512], bf16), T([8,1024,512], bf16), T([512], bf16), ...)
@oracle_impl(hardware="B200", point="40057a60", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        base_shape_arg,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
        shape6,
        shape7,
        shape8,
        shape9,
        shape10,
        shape11,
        shape12,
    ) = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    base_shape = tuple(int(dim) for dim in base_shape_arg)

    norm_base = torch.empty_strided(
        base_shape,
        _contiguous_stride(base_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]),)
    _residual_rmsnorm_bf16_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        norm_base,
        ROWS=rows,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
    )
    return (
        norm_base,
        norm_base.view(tuple(int(dim) for dim in shape1)),
        norm_base.view(tuple(int(dim) for dim in shape2)),
        norm_base.view(tuple(int(dim) for dim in shape3)),
        norm_base.view(tuple(int(dim) for dim in shape4)),
        norm_base.view(tuple(int(dim) for dim in shape5)),
        norm_base.view(tuple(int(dim) for dim in shape6)),
        norm_base.view(tuple(int(dim) for dim in shape7)),
        norm_base.view(tuple(int(dim) for dim in shape8)),
        norm_base.view(tuple(int(dim) for dim in shape9)),
        norm_base.view(tuple(int(dim) for dim in shape10)),
        norm_base.view(tuple(int(dim) for dim in shape11)),
        norm_base.view(tuple(int(dim) for dim in shape12)),
    )
