"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-J causal same-segment masked attention softmax scope in one Triton row kernel, including the advanced-index position/segment predicate, the observable f32 scalar zero and `[1,1,128,128]` mask-fill tensor, bf16 divide-by-16 score scaling, stable fp32 last-dimension amax/libdevice.exp/sum/div, explicit bf16 probability cast, final contiguous `[16,128,128]` view, and returned permute alias, whereas Inductor lowers the decomposed iota/unsqueeze/index/where mask construction, scale, reduction, cast, view, and permute as generic graph fragments; Inductor cannot do this today because its pattern library does not recognize GPT-J's generated causal same-segment predicate as the producer of a full attention-softmax envelope with observable mask and layout side outputs; the fix is NEW_PATTERN: add a guarded GPT-J masked attention-softmax lowering that recomputes the cheap structured predicate inside the row-softmax kernel and emits all sibling outputs directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _gptj_masked_scaled_softmax_kernel(
    positions_ptr,
    segments_ptr,
    scores_ptr,
    zero_ptr,
    where_ptr,
    out_f32_ptr,
    out_bf16_ptr,
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

    query = rows - (rows // q_len) * q_len
    q_pos = tl.load(positions_ptr + query, mask=row_mask, other=0)
    k_pos = tl.load(positions_ptr + cols, mask=col_mask, other=0)
    q_seg = tl.load(segments_ptr + q_pos, mask=row_mask, other=-1)
    k_seg = tl.load(segments_ptr + k_pos, mask=col_mask, other=-2)

    keep = (k_pos[None, :] <= q_pos[:, None]) & (k_seg[None, :] == q_seg[:, None])
    mask_bias = tl.where(keep & valid, 0.0, -3.4028234663852886e38)

    score_offsets = rows[:, None] * k_len + cols[None, :]
    scores = tl.load(scores_ptr + score_offsets, mask=valid, other=0.0).to(tl.float32)
    scaled = (scores * 0.0625).to(tl.bfloat16).to(tl.float32)
    masked_scores = scaled + mask_bias

    row_max = tl.max(masked_scores, axis=1)
    safe_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(masked_scores - safe_max[:, None])
    numer = tl.where(keep & valid, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    denom = tl.where(row_mask, denom, 1.0)
    probs = numer / denom[:, None]

    tl.store(out_f32_ptr + score_offsets, probs, mask=valid)
    tl.store(out_bf16_ptr + score_offsets, probs.to(tl.bfloat16), mask=valid)

    where_mask = rows[:, None] < q_len
    tl.store(where_ptr + query[:, None] * k_len + cols[None, :], mask_bias, mask=where_mask & col_mask[None, :])
    tl.store(zero_ptr, 0.0, mask=tl.program_id(0) == 0)


@oracle_impl(hardware="B200", point="508a0a1f", BLOCK_M=4, BLOCK_N=128, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1

    heads = int(arg2_1.shape[0])
    q_len = int(arg2_1.shape[1])
    k_len = int(arg2_1.shape[2])
    n_rows = heads * q_len

    zero = torch.empty_strided((), (), device=arg2_1.device, dtype=torch.float32)
    where = torch.empty_strided(
        (1, 1, q_len, k_len),
        (q_len * k_len, q_len * k_len, k_len, 1),
        device=arg2_1.device,
        dtype=torch.float32,
    )
    out_f32 = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_2),
        (heads * q_len * k_len, q_len * k_len, k_len, 1),
        device=arg2_1.device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_3),
        (q_len * k_len, k_len, 1),
        device=arg2_1.device,
        dtype=torch.bfloat16,
    )

    _gptj_masked_scaled_softmax_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        zero,
        where,
        out_f32,
        out_bf16,
        n_rows=n_rows,
        q_len=q_len,
        k_len=k_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return zero, where, out_f32, out_bf16, out_bf16.permute(0, 2, 1)
