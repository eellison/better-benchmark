"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-J attention softmax scope in one Triton row kernel, including the shape-param view, bf16 divide-by-16 scale, broadcast fp32 bias add, stable fp32 last-dimension amax/libdevice.exp/sum/div, returned f32 probability tensor, explicit bf16 probability cast, final view, and returned permute alias, whereas Inductor lowers the decomposed scale/add/amax/sub/exp/sum/div/cast/view/permute graph as generic reduction and layout-output work; Inductor cannot do this today because its pattern library does not recognize this scaled biased attention softmax with sibling f32 and bf16 view outputs as one full-scope row template; the fix is NEW_PATTERN: add a guarded scaled-biased attention-softmax lowering that emits the f32 and bf16 stores from one row reduction and preserves metadata-only view aliases."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _scaled_biased_softmax_kernel(
    x_ptr,
    bias_ptr,
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
    mask = row_mask[:, None] & col_mask[None, :]

    q = rows - (rows // q_len) * q_len
    offsets = rows[:, None] * k_len + cols[None, :]
    bias_offsets = q[:, None] * k_len + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0).to(tl.float32)
    scores = x * 0.0625 + bias
    scores = tl.where(mask, scores, -float("inf"))

    row_max = tl.max(scores, axis=1)
    safe_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(scores - safe_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    denom = tl.where(row_mask, denom, 1.0)
    probs = numer / denom[:, None]

    tl.store(out_f32_ptr + offsets, probs, mask=mask)
    tl.store(out_bf16_ptr + offsets, probs, mask=mask)


# c2b7b80f: (T([16, 128, 128], bf16), T([1, 1, 128, 128], f32), ...)
@oracle_impl(hardware="B200", point="c2b7b80f", BLOCK_M=4, BLOCK_N=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    heads = int(arg0_1.shape[0])
    q_len = int(arg0_1.shape[1])
    k_len = int(arg0_1.shape[2])
    n_rows = heads * q_len

    out_f32 = torch.empty_strided(
        (1, heads, q_len, k_len),
        (heads * q_len * k_len, q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_2),
        (q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(n_rows, BLOCK_M),)
    _scaled_biased_softmax_kernel[grid](
        arg0_1,
        arg1_1,
        out_f32,
        out_bf16,
        n_rows=n_rows,
        q_len=q_len,
        k_len=k_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )

    return out_f32, out_bf16, out_bf16.permute(0, 2, 1)
