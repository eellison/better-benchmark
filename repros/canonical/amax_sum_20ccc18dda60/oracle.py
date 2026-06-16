"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete TrOCR bf16 broadcast-add attention softmax in one Triton row kernel, including the `[1024,256,256] -> [64,16,256,256]` metadata view, `[64,1,256,256]` broadcast producer, fp32 score construction from bf16 inputs, stable last-dimension amax/libdevice.exp/sum/div, final bf16 probability cast, and returned contiguous `[1024,256,256]` view, whereas Inductor lowers the decomposed view/add/view/cast/amax/sub/exp/sum/div/cast graph through generic reduction scheduling; Inductor cannot do this today because its pattern library does not recognize this K=256 broadcast-bias bf16 softmax as one full-scope row template with the view-only and broadcast producers sunk into the reduction/store; the fix is NEW_PATTERN: add a guarded TrOCR-style broadcast-add row-softmax lowering that fuses score construction, stable row reductions, bf16 cast boundaries, and final view emission."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _broadcast_add_softmax_kernel(
    x_ptr,
    bias_ptr,
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
    N_ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    Q_LEN: tl.constexpr,
    K_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < N_ROWS
    col_mask = cols < K_LEN
    mask = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // Q_LEN
    batch = flat_bh // HEADS
    query = rows - flat_bh * Q_LEN

    x_offsets = flat_bh[:, None] * x_s0 + query[:, None] * x_s1 + cols[None, :] * x_s2
    bias_offsets = batch[:, None] * bias_s0 + query[:, None] * bias_s2 + cols[None, :] * bias_s3

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

    out_offsets = flat_bh[:, None] * out_s0 + query[:, None] * out_s1 + cols[None, :] * out_s2
    tl.store(out_ptr + out_offsets, probs.to(tl.bfloat16), mask=mask)


# 38831ee4: (T([1024,256,256], bf16), T([64,1,256,256], bf16), S([64,16,256,256]), S([1024,256,256]))
@oracle_impl(hardware="B200", point="38831ee4", BLOCK_M=8, BLOCK_N=256, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1 = inputs
    del _shape_param_0

    out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_1),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    n_rows = int(arg0_1.shape[0] * arg0_1.shape[1])

    _broadcast_add_softmax_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
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
        N_ROWS=n_rows,
        HEADS=16,
        Q_LEN=arg0_1.shape[1],
        K_LEN=arg0_1.shape[2],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
