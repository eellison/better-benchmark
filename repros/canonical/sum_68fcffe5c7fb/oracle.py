"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete XGLM bf16 attention softmax-backward row update in one Triton kernel, including the dropout-mask f32 scaling, `[512,128,128] -> [32,16,128,128]` score view, broadcast f32 attention bias add, `max(add, -FLT_MAX)` probability reconstruction with natural libdevice exp and captured row denominator, fp32 row product sum, exact `fma.rn.f32` epilogue, the `add == -FLT_MAX` half-gradient and `add < -FLT_MAX` zero exceptional cases, final bf16 cast, and returned contiguous `[512,128,128]` view, whereas Inductor lowers the producer, reduction, fma, exceptional-value predicates, cast, and metadata-only views through generic reduction scheduling around materialized intermediates; Inductor cannot do this today because its scheduler does not recognize this saved-attention-backward envelope as one full-scope row-reduction template with the broadcast score-bias producer, fma epilogue, exceptional floor handling, dtype boundary, and output layout preserved; the fix is SCHEDULER_FUSION: add a guarded attention-backward row template that sinks score reconstruction, probability reconstruction, dependent row reduction, fma, floor predicates, bf16 cast, and output store into one generated plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


DROPOUT_SCALE_F32 = 1.1111111640930176
F32_MIN = -3.4028234663852886e38


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fma_rn_f32(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _xglm_softmax_backward_kernel(
    grad_ptr,
    keep_ptr,
    score_ptr,
    bias_ptr,
    row_shift_ptr,
    row_denom_ptr,
    out_ptr,
    bias_s0: tl.constexpr,
    bias_s2: tl.constexpr,
    bias_s3: tl.constexpr,
    n_rows: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    DROPOUT_SCALE: tl.constexpr,
    FLOOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * k_len + cols[None, :]

    flat_bh = rows // q_len
    outer = flat_bh // heads
    query = rows - flat_bh * q_len
    bias_offsets = (
        outer[:, None] * bias_s0
        + query[:, None] * bias_s2
        + cols[None, :] * bias_s3
    )

    grad = tl.load(
        grad_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    keep = tl.load(
        keep_ptr + offsets,
        mask=mask,
        other=0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    score_base = tl.load(
        score_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + bias_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    row_shift = tl.load(
        row_shift_ptr + rows,
        mask=row_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    row_denom = tl.load(
        row_denom_ptr + rows,
        mask=row_mask,
        other=1.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    add = score_base + bias
    floored = tl.maximum(add, FLOOR)
    probs = libdevice.exp(floored - row_shift[:, None]) / row_denom[:, None]

    scale = tl.full((BLOCK_M, BLOCK_N), DROPOUT_SCALE, tl.float32)
    scaled_keep = _mul_rn_f32(keep, scale)
    scaled_grad = _mul_rn_f32(grad, scaled_keep)
    product = _mul_rn_f32(scaled_grad, probs)
    row_sum = tl.sum(tl.where(mask, product, 0.0), axis=1)[:, None].to(tl.float32)
    fma = _fma_rn_f32(-probs, row_sum, product)
    half = _mul_rn_f32(fma, 0.5)
    selected_eq = tl.where(add == FLOOR, half, fma)
    selected = tl.where(add < FLOOR, 0.0, selected_eq)

    tl.store(out_ptr + offsets, selected.to(tl.bfloat16), mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# 8da16745: XGLM train softmax backward, [32,16,128,128] viewed as [512,128,128].
@oracle_impl(hardware="B200", point="8da16745", BLOCK_M=2, BLOCK_N=128, num_warps=2, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        shape0,
        _shape1,
        _shape2,
        shape3,
    ) = inputs
    full_shape = tuple(int(dim) for dim in shape0)
    out_shape = tuple(int(dim) for dim in shape3)
    outer, heads, q_len, k_len = full_shape
    n_rows = outer * heads * q_len

    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _xglm_softmax_backward_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        out,
        bias_s0=arg3_1.stride(0),
        bias_s2=arg3_1.stride(2),
        bias_s3=arg3_1.stride(3),
        n_rows=n_rows,
        heads=heads,
        q_len=q_len,
        k_len=k_len,
        DROPOUT_SCALE=DROPOUT_SCALE_F32,
        FLOOR=F32_MIN,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
