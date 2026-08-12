"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-Neo bf16 masked-attention softmax scope in one Triton row kernel, including the sliced external bool attention mask, generated causal predicate, segment-equality predicate from the indexed `[32,128]` table, returned bf16 additive mask tensor, fp32 finite/min score masking, fp32 stable last-dimension amax/libdevice.exp/sum/div, final bf16 probability cast, expand, and contiguous returned view, whereas Inductor lowers the decomposed slice/view/where/iota/index/eq/bitwise_and/where/add/amax/sub/exp/sum/div/cast/expand/view graph through generic mask producers and reduction scheduling; Inductor cannot do this today because its pattern library does not recognize a GPT-Neo structured masked-attention softmax with an observable additive-mask side output; the fix is NEW_PATTERN: add a guarded masked-attention-softmax lowering that recomputes cheap predicates inside the row-softmax schedule, emits the required bf16 side mask directly, and sinks the final layout-only view into the store."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _gptneo_masked_softmax_kernel(
    attn_mask_ptr,
    scores_ptr,
    index_ptr,
    add_mask_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    heads: tl.constexpr,
    seq_len: tl.constexpr,
    full_mask_stride: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < seq_len
    mask = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // seq_len
    batch = flat_bh // heads
    head = flat_bh - batch * heads
    query = rows - flat_bh * seq_len
    offsets = rows[:, None] * seq_len + cols[None, :]

    q_segment = tl.load(
        index_ptr + batch * seq_len + query,
        mask=row_mask,
        other=0,
    )
    k_segment = tl.load(
        index_ptr + batch[:, None] * seq_len + cols[None, :],
        mask=mask,
        other=q_segment[:, None] + 1,
    )
    causal = cols[None, :] <= query[:, None]
    same_segment = k_segment == q_segment[:, None]
    local_keep = causal & same_segment

    min_bf16 = tl.full((BLOCK_M, BLOCK_N), -3.3895313892515355e38, tl.float32).to(tl.bfloat16)
    add_mask = tl.where(local_keep, 0.0, min_bf16).to(tl.bfloat16)

    side_offsets = batch[:, None] * (seq_len * seq_len) + query[:, None] * seq_len + cols[None, :]
    tl.store(
        add_mask_ptr + side_offsets,
        add_mask,
        mask=mask & (head[:, None] == 0),
    )

    external_keep = tl.load(
        attn_mask_ptr + query[:, None] * full_mask_stride + cols[None, :],
        mask=mask,
        other=0,
    ).to(tl.int1)
    score = tl.load(scores_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    min_f32 = tl.full((BLOCK_M, BLOCK_N), -3.4028234663852886e38, tl.float32)
    masked_score = tl.where(external_keep, score, min_f32)
    logits = tl.where(mask, masked_score + add_mask.to(tl.float32), -float("inf"))

    row_max = tl.max(logits, axis=1)
    numer = libdevice.exp(logits - row_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    tl.store(out_ptr + offsets, probs, mask=mask)


def _launch(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, shape0, shape1, shape2, shape3 = inputs
    del shape0, shape1, shape2

    batch = int(arg2_1.shape[0])
    seq_len = int(arg2_1.shape[1])
    heads = int(arg1_1.shape[0] // batch)
    n_rows = int(arg1_1.shape[0] * seq_len)

    add_mask = torch.empty_strided(
        (batch, 1, seq_len, seq_len),
        (seq_len * seq_len, seq_len * seq_len, seq_len, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        tuple(int(dim) for dim in shape3),
        tuple(int(stride) for stride in arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    _gptneo_masked_softmax_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        add_mask,
        out,
        n_rows=n_rows,
        heads=heads,
        seq_len=seq_len,
        full_mask_stride=arg0_1.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return add_mask, out


# 5911cf96: (T([1,1,2048,2048], b8), T([512,128,128], f32), T([32,128], i64), ...)
@oracle_impl(hardware="B200", point="5911cf96", BLOCK_M=8, BLOCK_N=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    return _launch(inputs, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=num_warps)
