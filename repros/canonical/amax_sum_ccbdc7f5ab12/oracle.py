"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-J causal same-segment masked attention softmax scope in one Triton row kernel, including the shape-param score view, fp32 divide-by-16 score scaling, advanced-index position/segment predicate, returned bf16 `[1,1,128,128]` mask-fill tensor, fp32 stable last-dimension amax/libdevice.exp/sum/div with the bf16 mask promotion preserved, explicit bf16 probability cast, and final contiguous `[16,128,128]` view, whereas Inductor lowers the decomposed iota/unsqueeze/index/where mask construction, scale, add, reduction, cast, expand, and view as generic graph fragments; Inductor cannot do this today because its pattern library does not recognize GPT-J's generated causal same-segment predicate as the producer of a full attention-softmax envelope with an observable bf16 mask output and layout-only epilogue; the fix is NEW_PATTERN: add a guarded GPT-J masked attention-softmax lowering that recomputes the structured predicate inside the row-softmax kernel and emits the sibling mask and bf16 probability view directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _gptj_masked_scaled_softmax_kernel(
    scores_ptr,
    segments_ptr,
    where_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    valid = row_mask[:, None] & col_mask[None, :]

    heads = rows // q_len
    query = rows - heads * q_len
    q_seg = tl.load(segments_ptr + query, mask=row_mask, other=-1)
    k_seg = tl.load(segments_ptr + cols, mask=col_mask, other=-2)
    keep = (cols[None, :] <= query[:, None]) & (k_seg[None, :] == q_seg[:, None])

    zero_bf16 = tl.full((BLOCK_M, BLOCK_N), 0.0, tl.float32).to(tl.bfloat16)
    min_bf16 = tl.full(
        (BLOCK_M, BLOCK_N),
        -3.3895313892515355e38,
        tl.float32,
    ).to(tl.bfloat16)
    mask_bf16 = tl.where(keep & valid, zero_bf16, min_bf16)

    offsets = rows[:, None] * k_len + cols[None, :]
    scores = tl.load(scores_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    masked_scores = scores * 0.0625 + mask_bf16.to(tl.float32)
    masked_scores = tl.where(valid, masked_scores, -float("inf"))

    row_max = tl.max(masked_scores, axis=1)
    safe_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(masked_scores - safe_max[:, None])
    numer = tl.where(keep & valid, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    denom = tl.where(row_mask, denom, 1.0)
    probs = numer / denom[:, None]

    tl.store(out_ptr + offsets, probs.to(tl.bfloat16), mask=valid)

    where_offsets = query[:, None] * k_len + cols[None, :]
    tl.store(
        where_ptr + where_offsets,
        mask_bf16,
        mask=(heads[:, None] == 0) & valid,
    )


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# 94a9fea0: (T([16,128,128], f32), T([1,128], i64), S([1,16,128,128]), ...)
@oracle_impl(hardware="B200", point="94a9fea0", BLOCK_M=8, BLOCK_N=128, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    out_shape = tuple(int(dim) for dim in _shape_param_3)
    q_len = int(arg0_1.shape[1])
    k_len = int(arg0_1.shape[2])
    n_rows = int(arg0_1.shape[0] * q_len)

    where = torch.empty_strided(
        (1, 1, q_len, k_len),
        (q_len * k_len, q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _gptj_masked_scaled_softmax_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        where,
        out,
        n_rows=n_rows,
        q_len=q_len,
        k_len=k_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return where, out
