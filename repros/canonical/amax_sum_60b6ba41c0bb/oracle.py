"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-Neo bf16 causal same-segment masked attention softmax scope in Triton, including the returned i64 iota, returned f32 and bf16 scalar fills, generated position-order and segment-equality predicate from the indexed `[1,128]` and `[32,128]` integer tables, returned f32 additive mask tensor, sliced external bool attention mask, bf16 score fill boundary, fp32 stable last-dimension amax/libdevice.exp/sum/div side outputs, final bf16 probability cast, contiguous `[512,128,128]` view, and returned permute alias, whereas Inductor lowers the decomposed iota/unsqueeze/index/bitwise_and/where/slice/where/add/amax/sub/exp/sum/div/cast/expand/view/permute graph through generic mask producers and reduction scheduling; Inductor cannot do this today because its attention-softmax pattern library does not recognize GPT-Neo's structured causal same-segment predicate with an observable f32 mask-bias side output and a separate bf16 external-mask score boundary; the fix is NEW_PATTERN: add a guarded GPT-Neo masked-attention softmax lowering that recomputes the cheap predicates inside the row-softmax kernel, emits the visible constants and mask side output, and sinks the final layout-only epilogue into the probability store."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _init_outputs_kernel(iota_ptr, zero_ptr, bf16_fill_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(iota_ptr + offsets, offsets)
    tl.store(zero_ptr, 0.0)
    fill = tl.full((), -3.3895313892515355e38, tl.float32).to(tl.bfloat16)
    tl.store(bf16_fill_ptr, fill)


@triton.jit
def _gptneo_masked_softmax_kernel(
    position_ptr,
    segment_ptr,
    scores_ptr,
    attn_mask_ptr,
    add_mask_ptr,
    amax_ptr,
    sum_ptr,
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
    active = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // seq_len
    batch = flat_bh // heads
    head = flat_bh - batch * heads
    query = rows - flat_bh * seq_len
    dense_offsets = rows[:, None] * seq_len + cols[None, :]

    q_pos = tl.load(position_ptr + query, mask=row_mask, other=0)
    k_pos = tl.load(position_ptr + cols, mask=col_mask, other=0)
    q_segment = tl.load(segment_ptr + batch * seq_len + q_pos, mask=row_mask, other=0)
    k_segment = tl.load(
        segment_ptr + batch[:, None] * seq_len + k_pos[None, :],
        mask=active,
        other=q_segment[:, None] + 1,
    )
    local_keep = (k_pos[None, :] <= q_pos[:, None]) & (k_segment == q_segment[:, None])

    min_f32 = tl.full((BLOCK_M, BLOCK_N), -3.4028234663852886e38, tl.float32)
    local_bias = tl.where(local_keep, 0.0, min_f32)
    side_offsets = batch[:, None] * (seq_len * seq_len) + query[:, None] * seq_len + cols[None, :]
    tl.store(add_mask_ptr + side_offsets, local_bias, mask=active & (head[:, None] == 0))

    external_keep = tl.load(
        attn_mask_ptr + query[:, None] * full_mask_stride + cols[None, :],
        mask=active,
        other=0,
    ).to(tl.int1)
    raw = tl.load(scores_ptr + dense_offsets, mask=active, other=0.0)
    bf16_fill = tl.full((BLOCK_M, BLOCK_N), -3.3895313892515355e38, tl.float32).to(tl.bfloat16)
    externally_masked = tl.where(external_keep, raw, bf16_fill)
    logits = tl.where(active, externally_masked.to(tl.float32) + local_bias, -float("inf"))

    row_max = tl.max(logits, axis=1)
    numer = libdevice.exp(logits - row_max[:, None])
    numer = tl.where(active, numer, 0.0)
    denom = tl.sum(numer.to(tl.float64), axis=1).to(tl.float32)
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    tl.store(amax_ptr + rows, row_max, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)
    tl.store(out_ptr + dense_offsets, probs, mask=active)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    known = 1
    missing = -1
    for idx, dim in enumerate(dims):
        if dim == -1:
            missing = idx
        else:
            known *= dim
    if missing >= 0:
        dims[missing] = int(numel) // known
    return tuple(dims)


@oracle_impl(hardware="B200", point="4459026d", BLOCK_M=8, BLOCK_N=128, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs
    view_shape = _shape_tuple(shape1)
    mask_shape = _resolve_shape(shape0, int(arg1_1.shape[0]) * int(view_shape[2]) * int(view_shape[3]))
    out_shape = _shape_tuple(shape3)
    row_shape = (view_shape[0], view_shape[1], view_shape[2], 1)

    iota = torch.empty_strided((32,), (1,), device=arg2_1.device, dtype=torch.int64)
    zero = torch.empty_strided((), (), device=arg2_1.device, dtype=torch.float32)
    bf16_fill = torch.empty_strided((), (), device=arg2_1.device, dtype=torch.bfloat16)
    add_mask = torch.empty_strided(
        mask_shape,
        _contiguous_stride(mask_shape),
        device=arg2_1.device,
        dtype=torch.float32,
    )
    amax = torch.empty_strided(
        row_shape,
        _contiguous_stride(row_shape),
        device=arg2_1.device,
        dtype=torch.float32,
    )
    sum_1 = torch.empty_strided(
        row_shape,
        _contiguous_stride(row_shape),
        device=arg2_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg2_1.device,
        dtype=torch.bfloat16,
    )

    _init_outputs_kernel[(1,)](
        iota,
        zero,
        bf16_fill,
        BLOCK=32,
        num_warps=1,
        num_stages=1,
    )

    heads = int(view_shape[1])
    seq_len = int(view_shape[2])
    n_rows = int(arg2_1.numel() // seq_len)
    _gptneo_masked_softmax_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        add_mask,
        amax,
        sum_1,
        out,
        n_rows=n_rows,
        heads=heads,
        seq_len=seq_len,
        full_mask_stride=arg3_1.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return iota, zero, add_mask, bf16_fill, amax, sum_1, out, out.permute(0, 2, 1)
