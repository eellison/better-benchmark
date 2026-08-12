"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MT5 causal relative-position attention softmax scope in one Triton row kernel, including the logarithmic relative-position bucket lookup, bf16 embedding-bias gather, causal bf16 minimum-fill mask, returned `[32,6,128,128]` bias/mask tensor with its head-inner stride layout, bf16-rounded score add, fp32 stable natural-exp softmax, final bf16 probability cast, expand, and contiguous `[192,128,128]` returned view, whereas Inductor lowers the decomposed iota/log/bucket/embedding/permute/mask/add/amax/sub/exp/sum/div/cast/expand/view graph through generic pointwise producers and reduction scheduling; Inductor cannot do this today because its pattern library does not canonicalize MT5 causal relative-position bucket construction plus an observable bias/mask side output into one full-scope attention-softmax plan; the fix is NEW_PATTERN: add a guarded MT5 relative-position attention-softmax lowering that recomputes structured buckets at point of use, emits the required side output in its requested layout, and sinks the layout-only epilogue into the row-softmax store."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _t5_causal_bucket(query, cols):
    distance = tl.where(cols <= query, query - cols, 0)
    bucket = distance
    bucket = tl.where(distance >= 16, 16, bucket)
    bucket = tl.where(distance >= 19, 17, bucket)
    bucket = tl.where(distance >= 21, 18, bucket)
    bucket = tl.where(distance >= 24, 19, bucket)
    bucket = tl.where(distance >= 27, 20, bucket)
    bucket = tl.where(distance >= 31, 21, bucket)
    bucket = tl.where(distance >= 35, 22, bucket)
    bucket = tl.where(distance >= 40, 23, bucket)
    bucket = tl.where(distance >= 46, 24, bucket)
    bucket = tl.where(distance >= 52, 25, bucket)
    bucket = tl.where(distance >= 59, 26, bucket)
    bucket = tl.where(distance >= 67, 27, bucket)
    bucket = tl.where(distance >= 77, 28, bucket)
    bucket = tl.where(distance >= 87, 29, bucket)
    bucket = tl.where(distance >= 99, 30, bucket)
    bucket = tl.where(distance >= 113, 31, bucket)
    return bucket


@triton.jit
def _relative_position_softmax_kernel(
    x_ptr,
    rel_bias_ptr,
    bias_out_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    Q_LEN: tl.constexpr,
    K_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_K)
    row_mask = rows < ROWS
    col_mask = cols < K_LEN
    active = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // Q_LEN
    batch = flat_bh // HEADS
    head = flat_bh - batch * HEADS
    query = rows - flat_bh * Q_LEN
    row_offsets = rows[:, None] * K_LEN + cols[None, :]

    bucket = _t5_causal_bucket(query[:, None], cols[None, :])
    bias = tl.load(
        rel_bias_ptr + bucket * HEADS + head[:, None],
        mask=active,
        other=0.0,
    )
    causal = cols[None, :] <= query[:, None]
    fill = tl.where(causal, 0.0, -3.3895313892515355e38)
    bias_mask = (bias.to(tl.float32) + fill).to(tl.bfloat16)

    bias_offsets = (
        batch[:, None] * (HEADS * Q_LEN * K_LEN)
        + head[:, None]
        + query[:, None] * (K_LEN * HEADS)
        + cols[None, :] * HEADS
    )
    tl.store(bias_out_ptr + bias_offsets, bias_mask, mask=active)

    x = tl.load(x_ptr + row_offsets, mask=active, other=0.0).to(tl.float32)
    scores = x + bias_mask.to(tl.float32)
    scores = tl.where(active, scores, -float("inf"))

    row_max = tl.max(scores, axis=1)
    row_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(scores - row_max[:, None])
    numer = tl.where(active, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    tl.store(out_ptr + row_offsets, probs, mask=active)


def _as_shape(shape):
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


def _launch(inputs, *, block_m: int, block_k: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, shape0, _shape1, _shape2, _shape3, shape4, shape5 = inputs
    del _shape1, _shape2, _shape3, shape4

    view_shape = _resolve_shape(shape0, arg0_1.numel())
    flat_shape = _resolve_shape(shape5, arg0_1.numel())
    batch = int(view_shape[0])
    heads = int(view_shape[1])
    q_len = int(view_shape[2])
    k_len = int(view_shape[3])
    rows = int(batch * heads * q_len)

    bias = torch.empty_strided(
        view_shape,
        (heads * q_len * k_len, 1, heads * k_len, heads),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        flat_shape,
        (q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _relative_position_softmax_kernel[(triton.cdiv(rows, block_m),)](
        arg0_1,
        arg1_1,
        bias,
        out,
        ROWS=rows,
        HEADS=heads,
        Q_LEN=q_len,
        K_LEN=k_len,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return bias, out


# e7595a1e: MT5 causal relative-position attention softmax, B=32, H=6, S=128.
@oracle_impl(hardware="B200", point="e7595a1e", block_m=8, block_k=128, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    block_m: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
):
    return _launch(
        inputs,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
