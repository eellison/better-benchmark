"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete TrOCR train broadcast-bias attention softmax scope in one Triton row kernel, including the `[1024,256,256] -> [64,16,256,256]` metadata view, f32 `[64,1,256,256]` broadcast add with bf16 score promotion, stable last-dimension amax/libdevice.exp/sum/div, returned fp32 amax and sum side outputs, final bf16 probability cast, returned contiguous `[1024,256,256]` probability view, and returned permute alias, whereas Inductor lowers the decomposed view/add/view/amax/sub/exp/sum/div/cast/permute graph through generic pointwise, reduction, and layout scheduler fragments; Inductor cannot do this today because its row-softmax scheduler does not recognize this TrOCR broadcast-bias producer and keep the f32 score construction, observable reduction side outputs, bf16 probability materialization, and alias-only epilogue resident in one full-scope row plan; the fix is NEW_PATTERN: add a guarded TrOCR broadcast-bias row-softmax lowering that fuses score construction, stable row reductions, side-output stores, bf16 cast boundaries, and layout-only returns."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _broadcast_add_softmax_side_outputs_kernel(
    x_ptr,
    bias_ptr,
    amax_ptr,
    sum_ptr,
    out_ptr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    bias_s0: tl.constexpr,
    bias_s2: tl.constexpr,
    bias_s3: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    n_rows: tl.constexpr,
    n_heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // q_len
    batch = flat_bh // n_heads
    query = rows - flat_bh * q_len

    x_offsets = flat_bh[:, None] * x_s0 + query[:, None] * x_s1 + cols[None, :] * x_s2
    bias_offsets = batch[:, None] * bias_s0 + query[:, None] * bias_s2 + cols[None, :] * bias_s3
    out_offsets = rows[:, None] * k_len + cols[None, :]

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0).to(tl.float32)
    scores = x + bias
    scores = tl.where(mask, scores, -float("inf"))

    row_max = tl.max(scores, axis=1)
    safe_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(scores - safe_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    tl.store(amax_ptr + rows, row_max, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)
    tl.store(out_ptr + out_offsets, probs.to(tl.bfloat16), mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# 19ef62d0: (T([1024,256,256], bf16), T([64,1,256,256], f32), S([64,16,256,256]), S([1024,256,256]))
@oracle_impl(hardware="B200", point="19ef62d0", BLOCK_M=8, BLOCK_N=256, num_warps=8, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, view_shape_arg, out_shape_arg = inputs
    view_shape = _shape_tuple(view_shape_arg)
    out_shape = _shape_tuple(out_shape_arg)
    n_heads = int(view_shape[1])
    q_len = int(view_shape[2])
    k_len = int(view_shape[3])
    n_rows = int(arg0_1.numel() // k_len)
    row_shape = out_shape[:-1] + (1,)

    amax = torch.empty_strided(
        row_shape,
        _contiguous_stride(row_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    sum_1 = torch.empty_strided(
        row_shape,
        _contiguous_stride(row_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _broadcast_add_softmax_side_outputs_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        amax,
        sum_1,
        out,
        x_s0=arg0_1.stride(0),
        x_s1=arg0_1.stride(1),
        x_s2=arg0_1.stride(2),
        bias_s0=arg1_1.stride(0),
        bias_s2=arg1_1.stride(2),
        bias_s3=arg1_1.stride(3),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        n_rows=n_rows,
        n_heads=n_heads,
        q_len=q_len,
        k_len=k_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return amax, sum_1, out, out.permute(0, 2, 1)
