"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete TrOCR causal attention softmax scope in one Triton row kernel, including the generated `col <= row` bf16 `[64,1,256,256]` causal mask side output with exact finite minimum fill, the `[1024,256,256] -> [64,16,256,256]` metadata view, broadcast bf16 mask add with the captured bf16 rounding boundary, fp32 stable last-dimension amax/libdevice.exp/sum/div, and final bf16 contiguous `[1024,256,256]` probability output, whereas Inductor lowers the iota/unsqueeze/le/expand/where mask construction, bf16 add/view/cast, and row-softmax through generic pointwise and reduction scheduler fragments; Inductor cannot do this today because its attention softmax template does not recognize a generated finite causal mask side output and keep that producer, the bf16 score rounding boundary, and the probability epilogue inside one full-scope row plan; the fix is NEW_PATTERN: add a guarded TrOCR causal-mask row-softmax lowering that emits the mask side output directly and sinks the broadcast mask add into the softmax row kernel while preserving dtype boundaries and output strides."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BF16_MIN = -3.3895313892515355e38


@triton.jit
def _causal_mask_softmax_kernel(
    x_ptr,
    mask_out_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    n_heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    fill_value: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    active = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // q_len
    batch = flat_bh // n_heads
    head = flat_bh - batch * n_heads
    query = rows - flat_bh * q_len
    offsets = rows[:, None] * k_len + cols[None, :]
    causal = cols[None, :] <= query[:, None]
    mask_bf16 = tl.where(causal, 0.0, fill_value).to(tl.bfloat16)

    mask_offsets = batch[:, None] * (q_len * k_len) + query[:, None] * k_len + cols[None, :]
    tl.store(mask_out_ptr + mask_offsets, mask_bf16, mask=active & (head[:, None] == 0))

    x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    score_bf16 = (x + mask_bf16.to(tl.float32)).to(tl.bfloat16)
    scores = tl.where(active, score_bf16.to(tl.float32), -float("inf"))
    row_max = tl.max(scores, axis=1)
    has_nan = tl.sum(tl.where(scores != scores, 1, 0), axis=1) != 0
    row_max = tl.where(has_nan, float("nan"), row_max)
    safe_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(scores - safe_max[:, None])
    numer = tl.where(active, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    tl.store(out_ptr + offsets, probs.to(tl.bfloat16), mask=active)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _resolve_shape(shape, numel=None):
    dims = [int(dim) for dim in shape]
    if numel is None:
        return tuple(1 if dim == -1 else dim for dim in dims)
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


def _launch(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    arg0_1, shape0, shape1, shape2 = inputs
    view_shape = _resolve_shape(shape0, arg0_1.numel())
    mask_shape = _resolve_shape(shape1)
    out_shape = _resolve_shape(shape2, arg0_1.numel())
    n_rows = int(arg0_1.numel() // view_shape[-1])
    n_heads = int(view_shape[1])
    q_len = int(view_shape[2])
    k_len = int(view_shape[3])

    mask = torch.empty_strided(
        mask_shape,
        _contiguous_stride(mask_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _causal_mask_softmax_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        arg0_1,
        mask,
        out,
        n_rows=n_rows,
        n_heads=n_heads,
        q_len=q_len,
        k_len=k_len,
        fill_value=BF16_MIN,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return mask, out


# 33572d5f: (T([1024,256,256], bf16), S([64,16,256,256]), S([64,-1,256,256]), S([1024,256,256]))
@oracle_impl(hardware="B200", point="33572d5f", BLOCK_M=8, BLOCK_N=256, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    return _launch(
        inputs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
