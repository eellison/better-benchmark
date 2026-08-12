"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Visformer safe scaled bf16 attention softmax scope for the sliced 196x196 window, including Inductor's unscaled and scaled fp32 amax side outputs, the finite-row any guard, natural exp/sum/div, explicit bf16 probability cast, constant_pad_nd zero padding to [768,200,200], and returned compact permute view in one Triton row kernel, whereas Inductor lowers the decomposed slice/view/mul/cast/amax/finite-check/where/exp/sum/div/cast/pad/permute graph through generic pointwise, reduction, and padding kernels; Inductor cannot do this today because its softmax pattern library does not recognize a dual-max exceptional-value fallback with observable side outputs, bf16 rounding boundary, and a constant-pad/layout epilogue as one full-scope row template; the fix is NEW_PATTERN: add a guarded safe-scaled-softmax lowering that preserves the observed scaled-amax and bf16 cast semantics, emits the side amax/finite/sum outputs, and fuses the zero-padding plus returned layout view into one plan."""

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
def _visformer_safe_scaled_softmax_pad_kernel(
    x_ptr,
    raw_amax_ptr,
    scaled_amax_ptr,
    all_finite_ptr,
    sum_ptr,
    compact_ptr,
    padded_ptr,
    n_rows: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    input_stride_head: tl.constexpr,
    input_stride_q: tl.constexpr,
    padded_stride_head: tl.constexpr,
    padded_stride_q: tl.constexpr,
    padded_h: tl.constexpr,
    padded_w: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAD_RIGHT: tl.constexpr,
    PAD_BOTTOM: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]

    flat_head = rows // q_len
    query = rows - flat_head * q_len
    input_offsets = (
        flat_head[:, None] * input_stride_head
        + query[:, None] * input_stride_q
        + cols[None, :]
    )
    compact_offsets = rows[:, None] * k_len + cols[None, :]
    padded_offsets = (
        flat_head[:, None] * padded_stride_head
        + query[:, None] * padded_stride_q
        + cols[None, :]
    )

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
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    tl.store(raw_amax_ptr + rows, raw_max, mask=row_mask)
    tl.store(scaled_amax_ptr + rows, scaled_max, mask=row_mask)
    tl.store(all_finite_ptr + rows, all_finite, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)
    tl.store(compact_ptr + compact_offsets, probs, mask=mask)
    tl.store(padded_ptr + padded_offsets, probs, mask=mask)

    right_cols = tl.arange(0, PAD_RIGHT)
    right_offsets = (
        flat_head[:, None] * padded_stride_head
        + query[:, None] * padded_stride_q
        + (k_len + right_cols)[None, :]
    )
    tl.store(padded_ptr + right_offsets, 0.0, mask=row_mask[:, None])

    bottom_offsets = (
        flat_head[:, None] * padded_stride_head
        + (q_len + query)[:, None] * padded_stride_q
        + cols[None, :]
    )
    bottom_mask = row_mask[:, None] & (query[:, None] < PAD_BOTTOM) & (cols[None, :] < padded_w)
    tl.store(padded_ptr + bottom_offsets, 0.0, mask=bottom_mask)


# df5bbfd4: (T([768,200,200], bf16), S([128,6,196,196]), S([128,6,196,196]), S([768,196,196]), S([0,4,0,4,0,0]))
@oracle_impl(hardware="B200", point="df5bbfd4", BLOCK_M=8, BLOCK_N=256, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_1, _shape_param_3

    view_shape = tuple(int(dim) for dim in _shape_param_0)
    compact_shape = tuple(int(dim) for dim in _shape_param_2)
    padded_shape = tuple(int(dim) for dim in arg0_1.shape)
    q_len = compact_shape[1]
    k_len = compact_shape[2]
    n_rows = compact_shape[0] * q_len

    reduction_shape = (view_shape[0], view_shape[1], view_shape[2], 1)
    reduction_stride = (view_shape[1] * view_shape[2], view_shape[2], 1, 1)
    raw_amax = torch.empty_strided(
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
    compact = torch.empty_strided(
        compact_shape,
        (compact_shape[1] * compact_shape[2], compact_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    padded = torch.empty_strided(
        padded_shape,
        (padded_shape[1] * padded_shape[2], padded_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _visformer_safe_scaled_softmax_pad_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        arg0_1,
        raw_amax,
        scaled_amax,
        all_finite,
        denom,
        compact,
        padded,
        n_rows=n_rows,
        q_len=q_len,
        k_len=k_len,
        input_stride_head=arg0_1.stride(0),
        input_stride_q=arg0_1.stride(1),
        padded_stride_head=padded.stride(0),
        padded_stride_q=padded.stride(1),
        padded_h=padded_shape[1],
        padded_w=padded_shape[2],
        scale=0.125,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        PAD_RIGHT=padded_shape[2] - k_len,
        PAD_BOTTOM=padded_shape[1] - q_len,
        num_warps=num_warps,
        num_stages=3,
    )

    return raw_amax, scaled_amax, all_finite, denom, padded, compact.permute(0, 2, 1)
