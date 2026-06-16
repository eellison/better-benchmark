"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 selected-slice row L2 normalization tail from Repro.forward, returning the metadata-only `arg0[:, -1, :]` bf16 view, the unclamped f32 row norm, and the f32 normalized output from the clamped norm divisor. Inductor lowers the select, f32 square/sum/sqrt row reduction, clamp/expand, and dependent broadcast divide through generic reduction-plus-pointwise scheduling; it cannot do this today because its scheduler lacks a selected-slice vector-norm template that keeps the row norm in registers while producing the dependent full-row epilogue and preserving the returned input alias. The fix is SCHEDULER_FUSION: add an alias-aware row-normalization lowering for selected slices with a scalar reduction result and dependent broadcast division."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 64
TIMESTEPS = 50
COLS = 256
SELECTED_TIMESTEP = 49


@triton.jit
def _selected_row_norm_kernel(
    x_ptr,
    norm_ptr,
    div_ptr,
    INPUT_ROW_STRIDE: tl.constexpr,
    SELECT_OFFSET: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * INPUT_ROW_STRIDE + SELECT_OFFSET + cols[None, :]

    x = tl.load(x_ptr + offsets).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=1)
    norm = tl.sqrt(sum_sq)
    denom = tl.maximum(norm, 1.0e-12)
    out = x / denom[:, None]

    tl.store(norm_ptr + rows, norm)
    tl.store(div_ptr + rows[:, None] * N_COLS + cols[None, :], out)


# torchbench_tts_angular train, arg0 bf16[64,50,256] -> select/norm/div.
@oracle_impl(hardware="B200", point="6cbb208b", BLOCK_M=1, BLOCK_N=256, num_warps=1)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    x, shape_param = inputs
    selected = torch.as_strided(
        x,
        (BATCH, COLS),
        (TIMESTEPS * COLS, 1),
        storage_offset=SELECTED_TIMESTEP * COLS,
    )
    norm = torch.empty_strided((BATCH, 1), (1, 1), device=x.device, dtype=torch.float32)
    div = torch.empty_strided(
        tuple(int(dim) for dim in shape_param),
        (COLS, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _selected_row_norm_kernel[(triton.cdiv(BATCH, BLOCK_M),)](
        x,
        norm,
        div,
        INPUT_ROW_STRIDE=TIMESTEPS * COLS,
        SELECT_OFFSET=SELECTED_TIMESTEP * COLS,
        N_COLS=COLS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return selected, norm, div
