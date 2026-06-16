"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Blenderbot bf16 broadcast-mask attention-softmax scope in one Triton row kernel, including the stride-zero bool mask to 0/-inf bias, `[512,128,128]` to `[16,32,128,128]` view semantics, fp32 stable last-dimension amax/libdevice.exp/sum/div, explicit bf16 probability rounding, all-`-inf` row fallback to zeros, returned 4D tensor, and returned `[512,128,128]` alias view sharing the same storage, whereas Inductor lowers the decomposed where/view/add/cast/amax/sub/exp/sum/div/cast/eq/any/where/expand/view graph through generic reduction and pointwise scheduling; Inductor cannot do this today because its pattern library does not recognize this all-masked-safe bf16 broadcast-mask softmax with a sibling layout-only alias output as one semantic row lowering; the fix is NEW_PATTERN: add a masked attention-softmax lowering that fuses predicate handling, row reductions, bf16 rounding, zero-row fallback, and alias-view epilogue into one generated kernel."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _masked_bf16_softmax_kernel(
    scores_ptr,
    mask_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    Q_LEN: tl.constexpr,
    K_LEN: tl.constexpr,
    MASK_S0: tl.constexpr,
    MASK_S2: tl.constexpr,
    MASK_S3: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < ROWS
    col_mask = cols < K_LEN
    tile_mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * K_LEN + cols[None, :]

    flat_bh = rows // Q_LEN
    batch = flat_bh // HEADS
    q = rows - (rows // Q_LEN) * Q_LEN
    mask_offsets = batch[:, None] * MASK_S0 + q[:, None] * MASK_S2 + cols[None, :] * MASK_S3
    keep = tl.load(mask_ptr + mask_offsets, mask=tile_mask, other=0) != 0
    active = tile_mask & keep

    scores = tl.load(scores_ptr + offsets, mask=active, other=-float("inf")).to(tl.float32)
    row_max = tl.max(scores, axis=1)
    has_any = tl.max(tl.where(active, 1, 0), axis=1) != 0
    safe_max = tl.where(has_any, row_max, 0.0)
    numer = libdevice.exp(scores - safe_max[:, None])
    numer = tl.where(active, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    safe_denom = tl.where(has_any, denom, 1.0)
    probs = (numer / safe_denom[:, None]).to(tl.bfloat16)
    zeros = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32).to(tl.bfloat16)
    out = tl.where(has_any[:, None], probs, zeros)

    tl.store(out_ptr + offsets, out, mask=tile_mask)


# f414433f: bool mask [16,1,128,128] stride (0,128,1,0), bf16 scores [512,128,128].
@oracle_impl(hardware="B200", point="f414433f", BLOCK_M=4, BLOCK_N=128, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    flat_shape = tuple(int(dim) for dim in _shape_param_3)
    q_len = int(flat_shape[1])
    k_len = int(flat_shape[2])
    rows = int(arg1_1.numel() // k_len)
    flat_out = torch.empty_strided(
        flat_shape,
        (q_len * k_len, k_len, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    _masked_bf16_softmax_kernel[(triton.cdiv(rows, BLOCK_M),)](
        arg1_1,
        arg0_1,
        flat_out,
        ROWS=rows,
        HEADS=32,
        Q_LEN=q_len,
        K_LEN=k_len,
        MASK_S0=arg0_1.stride(0),
        MASK_S2=arg0_1.stride(2),
        MASK_S3=arg0_1.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return torch.as_strided(flat_out, (16, 32, q_len, k_len), (32 * q_len * k_len, q_len * k_len, k_len, 1)), flat_out
