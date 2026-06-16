"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full Whisper-tiny bf16 convolution GELU, `[1,384,1500] -> [1,1500,384]` layout-producing position add, clone-equivalent contiguous fp32 population `var_mean(..., dim=2, correction=0)` LayerNorm, affine epilogue, bf16 cast, visible non-contiguous add output, and three returned `[1500,384]` normalized view aliases in one Triton row kernel; Inductor already lowers this fixed-hidden norm-template scope close to the same memory-traffic envelope, but its generic schedule materializes the GELU/add producer before the normalization consumer and repeated view returns; the fix is BANDWIDTH_BOUND: record this as a full-scope at-floor Whisper GELU-position-LayerNorm case unless broader norm-template math-codegen or launch-overhead work moves the family."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _whisper_gelu_position_layernorm_kernel(
    x_ptr,
    position_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    SEQ_LEN: tl.constexpr,
    HIDDEN: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_S: tl.constexpr,
    ADD_STRIDE_S: tl.constexpr,
    ADD_STRIDE_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = rows[:, None] < SEQ_LEN
    col_mask = cols[None, :] < HIDDEN
    mask = row_mask & col_mask

    x_offsets = cols[None, :] * X_STRIDE_C + rows[:, None] * X_STRIDE_S
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    half = x * 0.5
    gelu = half * (tl.erf(x * 0.7071067811865476) + 1.0)
    gelu_bf16 = gelu.to(tl.bfloat16)

    position = tl.load(
        position_ptr + rows[:, None] * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    added_bf16 = (gelu_bf16 + position).to(tl.bfloat16)
    add_offsets = rows[:, None] * ADD_STRIDE_S + cols[None, :] * ADD_STRIDE_H
    tl.store(add_out_ptr + add_offsets, added_bf16, mask=mask)

    values = added_bf16.to(tl.float32)
    mean = tl.sum(tl.where(mask, values, 0.0), axis=1)[:, None] / HIDDEN
    centered = values - mean
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / HIDDEN
    invstd = libdevice.rsqrt(variance + 1.0e-5)

    weight = tl.load(
        weight_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    normalized = (centered * invstd) * weight[None, :] + bias[None, :]
    norm_offsets = rows[:, None] * HIDDEN + cols[None, :]
    tl.store(norm_out_ptr + norm_offsets, normalized.to(tl.bfloat16), mask=mask)


# e5172202: (T([1,384,1500], bf16), T([1500,384], bf16), T([384], bf16), T([384], bf16), S([1500,384]), S([1500,384]), S([1500,384]))
@oracle_impl(hardware="B200", point="e5172202", BLOCK_H=512, ROW_BLOCK=4, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int, num_stages: int):
    x, position, weight, bias, shape0, shape1, shape2 = inputs
    seq_len = int(position.shape[0])
    hidden = int(position.shape[1])
    view_shape0 = tuple(int(dim) for dim in shape0)
    view_shape1 = tuple(int(dim) for dim in shape1)
    view_shape2 = tuple(int(dim) for dim in shape2)

    add_out = torch.empty_strided(
        (1, seq_len, hidden),
        (seq_len * hidden, 1, seq_len),
        device=x.device,
        dtype=torch.bfloat16,
    )
    norm_base = torch.empty_strided(
        (1, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _whisper_gelu_position_layernorm_kernel[(triton.cdiv(seq_len, ROW_BLOCK),)](
        x,
        position,
        weight,
        bias,
        add_out,
        norm_base,
        SEQ_LEN=seq_len,
        HIDDEN=hidden,
        X_STRIDE_C=int(x.stride(1)),
        X_STRIDE_S=int(x.stride(2)),
        ADD_STRIDE_S=int(add_out.stride(1)),
        ADD_STRIDE_H=int(add_out.stride(2)),
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return (
        add_out,
        norm_base.view(view_shape0),
        norm_base.view(view_shape1),
        norm_base.view(view_shape2),
    )
