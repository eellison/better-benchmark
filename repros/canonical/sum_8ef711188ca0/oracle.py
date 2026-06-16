"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full bf16 full/select_scatter/view/permute plus dim-0 sum scope with one shape-specialized Triton kernel, recognizing that the singleton select_scatter exactly materializes the contiguous `[128, 768]` input as the backing storage for all three returned layout aliases while the same traversal can accumulate the sibling f32 column sums and apply the captured bf16 rounding boundary. Inductor lowers the mandatory materialized alias producer and the reduction through its generic multi-output reduction path, which cannot currently fuse this singleton scatter identity, alias-preserving layout store, and rounded side reduction into a compact fixed-extent schedule. The fix is SCHEDULER_FUSION: add a guarded multi-output reduction template for identity singleton select_scatter producers with returned aliases and compatible column reductions."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 128
COLS = 768


@triton.jit
def _materialize_sum_kernel(
    x_ptr,
    base_ptr,
    sum_ptr,
    ROWS_: tl.constexpr,
    COLS_: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, ROWS_)
    active = cols < COLS_
    offsets = rows[:, None] * COLS_ + cols[None, :]

    values = tl.load(x_ptr + offsets, mask=active[None, :], other=0.0)
    tl.store(base_ptr + offsets, values, mask=active[None, :])

    totals = tl.sum(values.to(tl.float32), axis=0)
    rounded = totals.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + cols, rounded, mask=active)


# timm_vit_base_patch16_siglip_256 train, bf16 [128,768] -> aliases plus rounded f32 [768].
@oracle_impl(hardware="B200", point="9c23c094", BLOCK_N=8, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    x, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    select_scatter = torch.empty_strided(
        (ROWS, 1, COLS),
        (COLS, COLS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided((COLS,), (1,), device=x.device, dtype=torch.float32)

    _materialize_sum_kernel[(triton.cdiv(COLS, BLOCK_N),)](
        x,
        select_scatter,
        sum_out,
        ROWS_=ROWS,
        COLS_=COLS,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )

    view = select_scatter.view(ROWS, COLS)
    return select_scatter, view, view.permute(1, 0), sum_out
