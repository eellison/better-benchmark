"""cuTile port of sum_20ba1158c4b0: TTS Angular scatter-reduce select_scatter."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 64
DEPTH = 50
COLS = 256
SCATTER_INDEX = DEPTH - 1


@ct.kernel
def _select_scatter_reduce_kernel(
    arg0_ptr,     # f32 [64, 256]
    arg1_ptr,     # f32 [64, 1]
    arg2_ptr,     # bf16 [64, 256]
    scatter_out_ptr,  # bf16 [64, 256] - the row to place at depth=49
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    arg0 = ct.load(arg0_ptr, index=(row, 0), shape=(1, BLOCK_N))
    arg1 = ct.load(arg1_ptr, index=(row, 0), shape=(1, 1))
    arg2 = ct.load(arg2_ptr, index=(row, 0), shape=(1, BLOCK_N))

    arg0_f = ct.astype(arg0, ct.float32)
    arg1_f = ct.astype(arg1, ct.float32)
    arg2_f = ct.astype(arg2, ct.float32)

    # denom = max(arg1, 1e-12) broadcast
    denom = ct.where(arg1_f < 1.0e-12, ct.full((1, 1), 1.0e-12, dtype=ct.float32), arg1_f)
    # arg2 / denom
    div_val = arg2_f / denom
    # neg = -arg0
    # row_terms = neg * (arg2/denom/denom) = -arg0 * (arg2/denom^2)
    row_terms = (-arg0_f) * (div_val / denom)
    row_sum = ct.sum(row_terms)

    # div2 = (arg0 / denom).to(bf16).to(f32)
    div2_bf16 = ct.astype(arg0_f / denom, ct.bfloat16)
    div2 = ct.astype(div2_bf16, ct.float32)

    # guarded_sum = row_sum if arg1 >= 1e-12 else 0
    # guarded_div = arg2/arg1 if arg1 != 0 else 0
    ge = arg1_f >= 1.0e-12  # (1,1)
    guarded_sum = ct.where(ge, row_sum, 0.0)
    raw_div = arg2_f / arg1_f
    eq_zero = arg1_f == 0.0
    guarded_div = ct.where(eq_zero, ct.full((1, 1), 0.0, dtype=ct.float32), raw_div)
    # mul1 = guarded_sum * guarded_div
    mul1_bf16 = ct.astype(guarded_sum * guarded_div, ct.bfloat16)
    scattered = ct.astype(div2 + ct.astype(mul1_bf16, ct.float32), ct.bfloat16)
    ct.store(scatter_out_ptr, index=(row, 0), tile=scattered)


@oracle_impl(hardware="B200", point="ee4b9eab", BLOCK_N=256)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1 = inputs
    device = arg0_1.device

    # Compute the scatter row via a cuTile kernel, then use PyTorch for the
    # zero-fill materialization and select_scatter (layout-only).
    scatter_row = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _select_scatter_reduce_kernel,
        (arg0_1, arg1_1, arg2_1, scatter_row, BLOCK_N),
    )

    out = torch.zeros((ROWS, DEPTH, COLS), device=device, dtype=torch.bfloat16)
    out[:, SCATTER_INDEX, :] = scatter_row
    return out
