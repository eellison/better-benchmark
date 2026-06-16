"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 MT5 residual-add RMSNorm scope in one Triton row kernel, including the metadata-only `[4096,512] -> [32,128,512]` view, bf16-rounded residual add, fp32 mean-square reduction with eps=1e-6, rsqrt normalization, bf16 normalization cast, bf16 affine weight multiply, and final contiguous `[4096,512]` view, whereas Inductor lowers the captured add/mean/rsqrt/affine/view graph through its generic fused reduction scheduler with the residual-add producer reloaded or recomputed for the normalization epilogue; Inductor cannot do this today because the row-reduction scheduler does not keep the bf16 residual-add tile live across the fixed-hidden RMS reduction while preserving the required low-precision rounding boundaries; the fix is SCHEDULER_FUSION: teach the norm-template scheduler to inline same-layout residual adds and the bf16 affine epilogue into one full-scope row plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _residual_rmsnorm_bf16_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
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
    tl.store(out_ptr + offsets, out)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# 841ed042: MT5 bf16[4096,512] residual-add RMSNorm -> bf16[4096,512]
@oracle_impl(
    hardware="B200",
    point="841ed042",
    BLOCK_H=512,
    BLOCK_M=1,
    num_warps=4,
    num_stages=2,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    BLOCK_M: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, base_shape_arg, out_shape_arg = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    base_shape = tuple(int(dim) for dim in base_shape_arg)
    out_shape = tuple(int(dim) for dim in out_shape_arg)

    out_base = torch.empty_strided(
        base_shape,
        _contiguous_stride(base_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _residual_rmsnorm_bf16_kernel[(triton.cdiv(rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        out_base,
        ROWS=rows,
        HIDDEN=hidden,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_base.view(out_shape)
