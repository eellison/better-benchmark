"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 view-to-f32 dim-0 column reduction as a shape-specialized Triton kernel that writes the final contiguous f32 `[1, 1, 768]` output directly, whereas Inductor lowers the metadata view, dtype conversion, and small static reduction through its generic reduction codegen path. Inductor cannot do this today because scheduler/codegen has no dedicated fixed-extent 128-row column-reduction template that strips generic reduction scaffolding while preserving the final keepdim layout. The fix is NEW_PATTERN: add a guarded small-column-reduction lowering for contiguous bf16 inputs reduced over the leading dimension."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 128
COLS = 768


@triton.jit
def _sum_rows_kernel(
    x_ptr,
    out_ptr,
    ROWS_: tl.constexpr,
    COLS_: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, ROWS_)
    active = cols < COLS_
    values = tl.load(
        x_ptr + rows[:, None] * COLS_ + cols[None, :],
        mask=active[None, :],
        other=0.0,
    ).to(tl.float32)
    totals = tl.sum(values, axis=0)
    tl.store(out_ptr + cols, totals, mask=active)


# timm_vit_base_patch16_siglip_256 train, bf16 [128,768] -> f32 [1,1,768].
@oracle_impl(hardware="B200", point="9c23c094", BLOCK_N=8, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    x, _shape_param = inputs
    out = torch.empty_strided(
        (1, 1, COLS),
        (COLS, COLS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    _sum_rows_kernel[(triton.cdiv(COLS, BLOCK_N),)](
        x,
        out,
        ROWS_=ROWS,
        COLS_=COLS,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
