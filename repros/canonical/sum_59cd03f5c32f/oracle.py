"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete TTS Angular selected-slice bf16 L2 normalization in one Triton row kernel, loading `arg0[:, -1, :]` from the original `[64,50,256]` input, converting the selected bf16 values to fp32 for the square/sum/sqrt reduction, applying the captured bf16 denominator cast before clamp/expand, and writing the final bf16 row-wise divide output, whereas Inductor lowers the select/convert/pow/sum/sqrt/cast/clamp/expand/div graph as generic reduction plus broadcast pointwise schedules; Inductor cannot do this today because its scheduler lacks a selected-slice row-normalization template that keeps the norm scalar resident while preserving the low-precision denominator and output rounding boundaries; the fix is SCHEDULER_FUSION: teach reduction scheduling to fuse selected-slice vector-norm reductions with their dependent broadcast division epilogue."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _selected_slice_l2_norm_kernel(
    x_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    STEP_STRIDE: tl.constexpr,
    SELECTED_STEP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)

    input_offsets = rows[:, None] * ROW_STRIDE + SELECTED_STEP * STEP_STRIDE + cols[None, :]
    selected = tl.load(x_ptr + input_offsets, mask=mask, other=0.0)
    selected_f32 = selected.to(tl.float32)

    sum_sq = tl.sum(selected_f32 * selected_f32, axis=1)
    denom_bf16 = tl.sqrt(sum_sq).to(tl.bfloat16, fp_downcast_rounding="rtne")
    denom = tl.maximum(denom_bf16.to(tl.float32), 1.0e-12)
    out = (selected_f32 / denom[:, None]).to(tl.bfloat16, fp_downcast_rounding="rtne")

    output_offsets = rows[:, None] * N + cols[None, :]
    tl.store(out_ptr + output_offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="6cbb208b", BLOCK_M=1, BLOCK_N=256, num_warps=1)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    arg0_1, _shape_param_0 = inputs
    out_shape = tuple(int(dim) for dim in _shape_param_0)
    rows = int(out_shape[0])
    cols = int(out_shape[1])

    out = torch.empty_strided(out_shape, (cols, 1), device=arg0_1.device, dtype=torch.bfloat16)
    _selected_slice_l2_norm_kernel[(triton.cdiv(rows, BLOCK_M),)](
        arg0_1,
        out,
        M=rows,
        N=cols,
        ROW_STRIDE=int(arg0_1.stride(0)),
        STEP_STRIDE=int(arg0_1.stride(1)),
        SELECTED_STEP=int(arg0_1.shape[1]) - 1,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
