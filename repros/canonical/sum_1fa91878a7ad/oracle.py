"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GPT-J bf16 attention-backward row epilogue in one Triton kernel, including the input view, bf16-to-fp32 cast, fp32 product, last-dimension sum, prims.fma-equivalent broadcast update, explicit bf16 rounding, final divide-by-16 rounding, and returned contiguous view, whereas Inductor lowers the captured cast/mul/sum/neg/fma/cast/div/view chain through its generic reduction scheduler; Inductor cannot do this today because the scheduler does not recognize this fixed-width row-reduction epilogue with an observable bf16 rounding boundary and scalar divide as one shape-specialized full-scope template; the fix is SCHEDULER_FUSION: add a guarded row-reduction epilogue template that sinks the view-only reshapes and preserves the f32 reduction, fused multiply-add, bf16 cast, scalar divide, and output layout."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _bf16_row_sum_fma_div_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < N_ROWS) & (cols[None, :] < N_COLS)
    offsets = rows[:, None] * N_COLS + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    product = x * y
    row_sum = tl.sum(tl.where(mask, product, 0.0), axis=1)[:, None]
    fma = tl.fma(-y, row_sum, product)
    rounded = fma.to(tl.bfloat16).to(tl.float32)
    out = (rounded * 0.0625).to(tl.bfloat16)

    tl.store(out_ptr + offsets, out, mask=mask)


# (T([16,128,128], bf16), T([1,16,128,128], f32), S([1,16,128,128]), S([16,128,128]))
@oracle_impl(hardware="B200", point="cdab8152", BLOCK_M=2, BLOCK_N=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N, num_warps):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1 = inputs
    out_shape = tuple(int(dim) for dim in _shape_param_1)
    n_cols = int(arg1_1.shape[-1])
    n_rows = arg1_1.numel() // n_cols

    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _bf16_row_sum_fma_div_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        out,
        N_ROWS=n_rows,
        N_COLS=n_cols,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
