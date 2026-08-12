"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 DistillGPT2 `[16384,50257]` view, padded transpose `[50264,16384]`, and right-padded row-major `[16384,50264]` output scope in one Triton tile kernel that loads each valid input element once and stores both returned padded materializations while zero-filling the seven-column vocabulary tail, whereas Inductor lowers the sibling transpose/pad and row-pad layout consumers through generic layout-copy scheduling; Inductor cannot do this today because its scheduler does not expose a reusable multi-output pad/layout plan for sibling consumers of the same large view with different affine store maps; the fix is SCHEDULER_FUSION: group these fixed zero-pad layout consumers and emit a fused multi-output tile kernel that preserves the fresh contiguous output contracts."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 16384
COLS = 50257
PAD = 7
PAD_COLS = COLS + PAD


@triton.jit
def _dual_pad_layout_kernel(
    input_ptr,
    out_transposed_ptr,
    out_rowmajor_ptr,
    ROWS_CONST: tl.constexpr,
    COLS_CONST: tl.constexpr,
    PAD_COLS_CONST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    in_bounds = (rows < ROWS_CONST) & (cols < COLS_CONST)
    out_bounds = (rows < ROWS_CONST) & (cols < PAD_COLS_CONST)

    values = tl.load(
        input_ptr + rows * COLS_CONST + cols,
        mask=in_bounds,
        other=0.0,
        eviction_policy="evict_last",
    )
    values = tl.where(cols < COLS_CONST, values, 0.0)

    tl.store(
        out_transposed_ptr + cols * ROWS_CONST + rows,
        values,
        mask=out_bounds,
    )
    tl.store(
        out_rowmajor_ptr + rows * PAD_COLS_CONST + cols,
        values,
        mask=out_bounds,
    )


def _launch(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    x, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    out_transposed = torch.empty_strided(
        (PAD_COLS, ROWS),
        (ROWS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    out_rowmajor = torch.empty_strided(
        (ROWS, PAD_COLS),
        (PAD_COLS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _dual_pad_layout_kernel[
        (triton.cdiv(ROWS, BLOCK_M), triton.cdiv(PAD_COLS, BLOCK_N))
    ](
        x,
        out_transposed,
        out_rowmajor,
        ROWS_CONST=ROWS,
        COLS_CONST=COLS,
        PAD_COLS_CONST=PAD_COLS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=4,
    )
    return out_transposed, out_rowmajor


# 846b3407: bf16[32,512,50257] -> bf16[50264,16384], bf16[16384,50264]
@oracle_impl(hardware="B200", point="846b3407", BLOCK_M=16, BLOCK_N=128, num_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
):
    return _launch(
        inputs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
