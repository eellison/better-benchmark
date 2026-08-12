"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete PLBart bf16 LM-head softmax-backward update, including ignore-index label handling, one-hot target-gradient algebra, the explicit bf16 rounding of the sparse target gradient, the explicit bf16 rounding before natural exp, the final bf16 residual add, and both returned padded layouts `[50008,16384]` and `[16384,50008]`. Inductor lowers the captured graph as generic one-hot expansion, dense pointwise materialization, sum, transpose, and padding work, so it expands the target mask over the whole vocabulary and schedules the two output layouts separately. It cannot do this today because scheduler/codegen does not recognize this label-softmax backward producer with observable bf16/f32 cast boundaries and sibling padded-layout returns as one full-scope template. The fix is SCHEDULER_FUSION: add a guarded LM-head backward materialization template that preserves the rounding boundaries, eliminates the dense one-hot producer algebraically, and emits both required padded output layouts from one producer traversal."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 16 * 1024
COLS = 50005
PADDED_COLS = 50008
PAD = PADDED_COLS - COLS


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _div_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _zero_pad_kernel(
    out_transposed_ptr,
    out_row_ptr,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    PADDED_N_COLS: tl.constexpr,
    PAD_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < (N_ROWS * PAD_SIZE)
    row = offsets // PAD_SIZE
    pad_col = offsets - row * PAD_SIZE
    zero = tl.zeros((BLOCK,), dtype=tl.bfloat16)

    tl.store(
        out_row_ptr + row * PADDED_N_COLS + N_COLS + pad_col,
        zero,
        mask=active,
    )
    tl.store(
        out_transposed_ptr + (N_COLS + pad_col) * N_ROWS + row,
        zero,
        mask=active,
    )


@triton.jit
def _plbart_lm_backward_kernel(
    numerator_ptr,
    denominator_ptr,
    labels_ptr,
    logits_ptr,
    row_shift_ptr,
    row_logsum_ptr,
    residual_ptr,
    out_transposed_ptr,
    out_row_ptr,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    LOGIT_ROW_STRIDE: tl.constexpr,
    PADDED_N_COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = row_offsets[:, None]
    cols = col_offsets[None, :]
    row_mask = row_offsets < N_ROWS
    active = row_mask[:, None] & (col_offsets[None, :] < N_COLS)

    scale = _div_rn_f32(
        tl.load(numerator_ptr).to(tl.float32),
        tl.load(denominator_ptr).to(tl.float32),
    )
    neg_scale = _mul_rn_f32(tl.full((), -1.0, tl.float32), scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)

    labels = tl.load(labels_ptr + row_offsets, mask=row_mask, other=-100)[:, None]
    in_range = (labels != -100) & (labels >= 0) & (labels < N_COLS)
    row_sum = tl.where(in_range, neg_scale, 0.0)
    target_grad = tl.where((cols == labels) & in_range, neg_scale, 0.0)

    logits = tl.load(
        logits_ptr + rows * LOGIT_ROW_STRIDE + cols,
        mask=active,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    row_shift = tl.load(row_shift_ptr + row_offsets, mask=row_mask, other=0.0)[:, None].to(
        tl.float32
    )
    row_logsum = tl.load(row_logsum_ptr + row_offsets, mask=row_mask, other=0.0)[
        :, None
    ].to(tl.float32)

    shifted = _sub_rn_f32(_sub_rn_f32(logits, row_shift), row_logsum).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    probs = libdevice.exp(shifted.to(tl.float32))
    softmax_grad = _sub_rn_f32(target_grad, _mul_rn_f32(probs, row_sum)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    residual = tl.load(
        residual_ptr + rows * N_COLS + cols,
        mask=active,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    out = _add_rn_f32(residual, softmax_grad.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(
        out_row_ptr + rows * PADDED_N_COLS + cols,
        out,
        mask=active,
    )
    tl.store(
        out_transposed_ptr + cols * N_ROWS + rows,
        out,
        mask=active,
    )


# b73ba80f: PLBart train LM-head softmax backward, rows=16384 vocab=50005.
@oracle_impl(
    hardware="B200",
    point="b73ba80f",
    BLOCK_M=16,
    BLOCK_N=256,
    PAD_BLOCK=1024,
    num_warps=8,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    PAD_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        *_,
    ) = inputs
    device = arg3_1.device

    out_transposed = torch.empty_strided(
        (PADDED_COLS, ROWS),
        (ROWS, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_row = torch.empty_strided(
        (ROWS, PADDED_COLS),
        (PADDED_COLS, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    _zero_pad_kernel[(triton.cdiv(ROWS * PAD, PAD_BLOCK),)](
        out_transposed,
        out_row,
        N_ROWS=ROWS,
        N_COLS=COLS,
        PADDED_N_COLS=PADDED_COLS,
        PAD_SIZE=PAD,
        BLOCK=PAD_BLOCK,
        num_warps=4,
    )
    _plbart_lm_backward_kernel[(triton.cdiv(ROWS, BLOCK_M), triton.cdiv(COLS, BLOCK_N))](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        out_transposed,
        out_row,
        N_ROWS=ROWS,
        N_COLS=COLS,
        LOGIT_ROW_STRIDE=PADDED_COLS,
        PADDED_N_COLS=PADDED_COLS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_transposed, out_row
