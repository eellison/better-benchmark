"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete dense CrossEntropyBackward output while deriving the sum over the expanded one-hot target row directly from the label and scalar bf16 gradient, including the `label == -100` ignore-index behavior, the exact f32 subtraction order `logits - row_shift0 - row_shift1`, the bf16 round-trip before natural libdevice exp, and the final bf16 output store, whereas Inductor lowers the iota/expand/equality one-hot construction, row reduction, dense exp epilogue, and bf16 cast as generic full-matrix producers over `[8192,262144]`; Inductor cannot do this today because it does not recognize the one-hot equality sum as a closed-form indexed scalar feeding the same full-row output epilogue; the fix is ALGEBRAIC_ELIMINATION: add a scheduler/FX rewrite that eliminates the expanded one-hot reduction and keeps only the label-indexed scalar row summary plus the dense faithful epilogue."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _xent_backward_epilogue_kernel(
    logits_ptr,
    shift0_ptr,
    shift1_ptr,
    labels_ptr,
    grad_scalar_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    col_mask = cols < n_cols
    offsets = rows * n_cols + cols

    labels = tl.load(labels_ptr + rows)
    grad = tl.load(grad_scalar_ptr).to(tl.float32)
    valid = (labels != -100) & (labels >= 0) & (labels < n_cols)
    row_sum = tl.where(valid, -grad, 0.0)
    sparse = tl.where((cols == labels) & valid, -grad, 0.0)

    logits = tl.load(logits_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
    shift0 = tl.load(shift0_ptr + rows).to(tl.float32)
    shift1 = tl.load(shift1_ptr + rows).to(tl.float32)
    shifted = logits - shift0
    shifted = shifted - shift1
    rounded = shifted.to(tl.bfloat16).to(tl.float32)
    dense = libdevice.exp(rounded) * row_sum
    out = sparse - dense
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=col_mask)


# ed4a60c9: genai CrossEntropyBackward static, logits bf16[8192,262144].
@oracle_impl(hardware="B200", point="ed4a60c9", BLOCK_M=1, BLOCK_N=2048, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    logits, shift0, shift1, labels, grad_scalar, _shape0, _shape1, _shape2 = inputs
    out_shape = tuple(int(dim) for dim in logits.shape)
    n_rows = int(out_shape[0])
    n_cols = int(out_shape[1])
    out = torch.empty_strided(
        out_shape,
        (n_cols, 1),
        device=logits.device,
        dtype=torch.bfloat16,
    )
    _xent_backward_epilogue_kernel[(triton.cdiv(n_rows, BLOCK_M), triton.cdiv(n_cols, BLOCK_N))](
        logits,
        shift0,
        shift1,
        labels,
        grad_scalar,
        out,
        n_cols=n_cols,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
