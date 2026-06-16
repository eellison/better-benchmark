"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Visformer safe scaled bf16 attention softmax scope, including the sliced 56-wide input layout, Inductor's fp32 scaled-amax return, unscaled row amax output, finite-row any guard, natural exp/sum/div, explicit bf16 probability cast, compact view output, and returned permute alias in one Triton row kernel, whereas Inductor lowers the decomposed slice/view/mul/cast/amax/finite-check/where/exp/sum/div/cast/view/permute graph through generic pointwise and reduction kernels; Inductor cannot do this today because its softmax pattern library does not recognize a dual-max exceptional-value fallback with observable side outputs and a sliced input layout as one full-scope row template; the fix is NEW_PATTERN: add a guarded safe-scaled-softmax lowering that preserves Inductor's observed scaled-amax lowering and bf16 probability rounding boundary, emits the side amax/finite/sum outputs, and returns metadata aliases from one fused plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _visformer_safe_scaled_softmax_kernel(
    x_ptr,
    amax_ptr,
    scaled_amax_ptr,
    all_finite_ptr,
    sum_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    k_len: tl.constexpr,
    input_stride_head: tl.constexpr,
    input_stride_q: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]

    flat_head = rows // 49
    query = rows - flat_head * 49
    input_offsets = (
        flat_head[:, None] * input_stride_head
        + query[:, None] * input_stride_q
        + cols[None, :]
    )
    compact_offsets = rows[:, None] * k_len + cols[None, :]

    raw = tl.load(x_ptr + input_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    raw = tl.where(col_mask[None, :], raw, -float("inf"))

    scaled = _f32_mul(raw, scale)

    raw_max = tl.max(raw, axis=1)
    scaled_max = tl.max(scaled, axis=1)

    finite = (scaled == scaled) & (tl.abs(scaled) != float("inf"))
    any_invalid = tl.max(tl.where(mask & ~finite, 1, 0), axis=1) != 0
    all_finite = ~any_invalid

    shifted_unscaled = _f32_mul(raw - raw_max[:, None], scale)
    shifted_scaled = scaled - scaled_max[:, None]
    shifted = tl.where(all_finite[:, None], shifted_unscaled, shifted_scaled)

    numer = libdevice.exp(shifted)
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    tl.store(amax_ptr + rows, raw_max, mask=row_mask)
    tl.store(scaled_amax_ptr + rows, scaled_max, mask=row_mask)
    tl.store(all_finite_ptr + rows, all_finite, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)
    tl.store(out_ptr + compact_offsets, probs.to(tl.bfloat16), mask=mask)


# 866d908a: (T([768,56,56], bf16), S([128,6,49,49]), S([128,6,49,49]), S([768,49,49]))
@oracle_impl(hardware="B200", point="866d908a", BLOCK_M=4, BLOCK_N=64, num_warps=1)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    out_shape = tuple(int(dim) for dim in _shape_param_2)
    n_rows = out_shape[0] * out_shape[1]
    k_len = out_shape[2]

    reduction_shape = (128, 6, 49, 1)
    reduction_stride = (294, 49, 1, 1)
    amax = torch.empty_strided(
        reduction_shape,
        reduction_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    scaled_amax = torch.empty_strided(
        reduction_shape,
        reduction_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    all_finite = torch.empty_strided(
        reduction_shape,
        reduction_stride,
        device=arg0_1.device,
        dtype=torch.bool,
    )
    denom = torch.empty_strided(
        reduction_shape,
        reduction_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _visformer_safe_scaled_softmax_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        arg0_1,
        amax,
        scaled_amax,
        all_finite,
        denom,
        out,
        n_rows=n_rows,
        k_len=k_len,
        input_stride_head=arg0_1.stride(0),
        input_stride_q=arg0_1.stride(1),
        scale=0.08838834764831845,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )

    return amax, scaled_amax, all_finite, denom, out, out.permute(0, 2, 1)
