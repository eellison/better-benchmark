"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Lennard-Jones bf16-cast outer-broadcast affine tanh scope with one Triton tile over the `[128, 16]` result, hoisting the 128 row operand and 16 column operands, preserving every f32->bf16->f32 rounding boundary, writing the visible bf16 `[128, 1]` cast of `arg2`, and materializing the bf16 tanh output directly. Inductor lowers the same graph as generic flattened pointwise code around a permute and broadcast maps, so it reloads broadcast operands per output element and emits the scalar multiply-by-one identities instead of selecting a small outer-product template. The fix is NEW_PATTERN: add a guarded tiled broadcast-pointwise lowering that hoists row/column loads, folds identity multiplies, and preserves the explicit bf16 cast boundaries plus sibling cast output."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 128
COLS = 16


@triton.jit
def _bf16_outer_tanh_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg2_bf16_out_ptr,
    tanh_out_ptr,
    ROWS_: tl.constexpr,
    COLS_: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, COLS_)
    row_mask = rows < ROWS_

    arg2_bf16 = tl.load(arg2_ptr + rows, mask=row_mask, other=0.0).to(tl.bfloat16)
    tl.store(arg2_bf16_out_ptr + rows, arg2_bf16, mask=row_mask)

    arg1_bf16 = tl.load(arg1_ptr + cols).to(tl.bfloat16)
    arg0_bf16 = tl.load(arg0_ptr + cols).to(tl.bfloat16)

    product = arg2_bf16[:, None].to(tl.float32) * arg1_bf16[None, :].to(tl.float32)
    affine = product + arg0_bf16[None, :].to(tl.float32)
    affine_bf16 = affine.to(tl.bfloat16)
    result = libdevice.tanh(affine_bf16.to(tl.float32))
    offsets = rows[:, None] * COLS_ + cols[None, :]
    tl.store(tanh_out_ptr + offsets, result, mask=row_mask[:, None])


# 1b1f1ebc: (T([16], f32), T([16,1], f32), T([128,1], f32))
@oracle_impl(hardware="B200", point="1b1f1ebc", BLOCK_M=8, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, num_warps: int):
    arg0_1, arg1_1, arg2_1 = inputs
    arg2_bf16 = torch.empty_strided(
        tuple(arg2_1.shape),
        tuple(arg2_1.stride()),
        device=arg2_1.device,
        dtype=torch.bfloat16,
    )
    tanh = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=arg2_1.device,
        dtype=torch.bfloat16,
    )

    _bf16_outer_tanh_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg2_bf16,
        tanh,
        ROWS_=ROWS,
        COLS_=COLS,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
        num_stages=3,
    )
    return arg2_bf16, tanh
