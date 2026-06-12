"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 GPT-J/GPT-Neo tanh-GELU backward expression, returns the dense bf16 tensor and its metadata-only transpose alias, and accumulates the sibling fp32 column sum from the bf16-rounded dense values in a fused Triton reduction plan, whereas Inductor currently lowers the returned pointwise producer, transpose view, and dependent sum through generic scheduler choices; Inductor cannot do this today because its scheduler/codegen does not keep a returned bf16 pointwise producer fused with a reduction consumer while preserving the explicit bf16 rounding boundaries and multi-output aliasing; the fix is SCHEDULER_FUSION: add a full-scope pointwise-plus-reduction template that stores the returned tensor once, reuses the rounded values for the reduction, and returns metadata views from the same storage."""

import torch
import triton
import triton.language as tl
from oracle_harness import oracle_impl
from triton.language.extra import libdevice


@triton.jit
def _gelu_backward_bf16(grad, x):
    grad_f32 = grad.to(tl.float32)
    x_f32 = x.to(tl.float32)
    x_half = (x_f32 * 0.5).to(tl.bfloat16).to(tl.float32)
    grad_x_half = grad_f32 * x_half

    x_cubed = x_f32 * x_f32 * x_f32
    tanh_arg = (x_f32 + x_cubed * 0.044715) * 0.7978845608028654
    tanh_val = libdevice.tanh(tanh_arg)

    gelu_base = (grad_f32 * (tanh_val + 1.0)).to(tl.bfloat16).to(tl.float32)
    sech2 = 1.0 - tanh_val * tanh_val
    scaled = grad_x_half * sech2 * 0.7978845608028654
    scaled_bf16 = scaled.to(tl.bfloat16).to(tl.float32)
    cubic_term = (scaled * 0.044715) * (x_f32 * x_f32 * 3.0)
    cubic_bf16 = cubic_term.to(tl.bfloat16).to(tl.float32)
    add_terms = (scaled_bf16 + cubic_bf16).to(tl.bfloat16).to(tl.float32)
    half_base = (gelu_base * 0.5).to(tl.bfloat16).to(tl.float32)
    return (add_terms + half_base).to(tl.bfloat16)


@triton.jit
def _single_pass_kernel(
    grad_ptr,
    x_ptr,
    out_ptr,
    sum_ptr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, BLOCK_M)
    offsets = rows[:, None] * N + cols[None, :]
    mask = cols[None, :] < N

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out_bf16 = _gelu_backward_bf16(grad, x)

    tl.store(out_ptr + offsets, out_bf16, mask=mask)
    col_sum = tl.sum(out_bf16.to(tl.float32), axis=0)
    rounded_sum = col_sum.to(tl.bfloat16).to(tl.float32)
    tl.store(sum_ptr + cols, rounded_sum, mask=cols < N)


@triton.jit
def _partial_kernel(
    grad_ptr,
    x_ptr,
    out_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m_tile = tl.program_id(0)
    n_tile = tl.program_id(1)
    rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out_bf16 = _gelu_backward_bf16(grad, x)

    tl.store(out_ptr + offsets, out_bf16, mask=mask)
    active_values = tl.where(mask, out_bf16.to(tl.float32), 0.0)
    partial = tl.sum(active_values, axis=0)
    partial_offsets = m_tile * N + cols
    tl.store(partial_ptr + partial_offsets, partial, mask=cols < N)


@triton.jit
def _finalize_kernel(
    partial_ptr,
    sum_ptr,
    N: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    tiles = tl.arange(0, BLOCK_TILES)
    offsets = tiles[:, None] * N + cols[None, :]
    mask = (tiles[:, None] < NUM_M_TILES) & (cols[None, :] < N)
    total = tl.sum(
        tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    rounded_total = total.to(tl.bfloat16).to(tl.float32)
    tl.store(sum_ptr + cols, rounded_total, mask=cols < N)


@oracle_impl(hardware="B200", point="3bd04781", BLOCK_M=128, BLOCK_N=16, num_warps=8)
@oracle_impl(hardware="B200", point="e036773d", BLOCK_M=32, BLOCK_N=32, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N, num_warps):
    grad, x, _shape_param_0, _shape_param_1, shape_out, shape_sum = inputs
    m = int(shape_out[0])
    n = int(shape_out[1])

    out = torch.empty_strided(
        tuple(shape_out),
        (n, 1),
        device=grad.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty(tuple(shape_sum), device=grad.device, dtype=torch.float32)

    if m == BLOCK_M:
        grid = (triton.cdiv(n, BLOCK_N),)
        _single_pass_kernel[grid](
            grad,
            x,
            out,
            sum_out,
            N=n,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
    else:
        num_m_tiles = triton.cdiv(m, BLOCK_M)
        partial = torch.empty((num_m_tiles, n), device=grad.device, dtype=torch.float32)
        grid = (num_m_tiles, triton.cdiv(n, BLOCK_N))
        _partial_kernel[grid](
            grad,
            x,
            out,
            partial,
            M=m,
            N=n,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        final_grid = (triton.cdiv(n, BLOCK_N),)
        _finalize_kernel[final_grid](
            partial,
            sum_out,
            N=n,
            NUM_M_TILES=num_m_tiles,
            BLOCK_TILES=triton.next_power_of_2(num_m_tiles),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )

    return out, out.permute(1, 0), sum_out
