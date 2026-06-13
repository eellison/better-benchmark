"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 residual-add RMSNorm scope in one shape-specialized Triton row kernel, including the metadata-only input view, bf16 residual-add rounding, fp32 mean-square reduction, eps=1e-6 rsqrt normalization, bf16 normalized-activation rounding, bf16 weight multiply, and final contiguous bf16 [1000,H] view, whereas Inductor lowers the same add/mean/rsqrt/affine/view graph through a generic reduction schedule that reloads or recomputes the residual-add producer for the normalization epilogue; Inductor cannot do this today because the row-reduction scheduler does not keep the full hidden tile's rounded residual producer live across the reduction while preserving the required bf16 cast boundaries; the fix is SCHEDULER_FUSION: add a guarded residual-RMSNorm row schedule that retains the rounded add tile through the reduction and emits the bf16 affine epilogue directly into the final view layout."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _residual_rmsnorm_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    offsets = row * HIDDEN + cols

    flat = tl.load(flat_ptr + offsets).to(tl.float32)
    residual = tl.load(residual_ptr + offsets).to(tl.float32)
    add_bf16 = (residual + flat).to(tl.bfloat16)

    x = add_bf16.to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(sum_sq * (1.0 / HIDDEN) + 1.0e-6)
    norm_bf16 = (x * inv_rms).to(tl.bfloat16)

    weight = tl.load(weight_ptr + cols).to(tl.float32)
    out = (norm_bf16.to(tl.float32) * weight).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out)


# da8b94aa: Qwen3, bf16 [1000,1024] residual RMSNorm -> bf16 [1000,1024]
@oracle_impl(hardware="B200", point="da8b94aa", BLOCK_H=1024, num_warps=4)
# 111af936: Mistral, bf16 [1000,4096] residual RMSNorm -> bf16 [1000,4096]
@oracle_impl(hardware="B200", point="111af936", BLOCK_H=4096, num_warps=4)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int):
    flat, residual, weight, _view_shape, out_shape = inputs
    rows = int(flat.shape[0])
    hidden = int(flat.shape[1])
    out = torch.empty_strided(
        tuple(int(dim) for dim in out_shape),
        (hidden, 1),
        device=flat.device,
        dtype=torch.bfloat16,
    )
    _residual_rmsnorm_kernel[(rows,)](
        flat,
        residual,
        weight,
        out,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=2,
    )
    return out
