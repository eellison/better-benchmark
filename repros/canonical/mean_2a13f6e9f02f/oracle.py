"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete SqueezeNet bf16 ReLU, fixed 13x13 spatial mean, final contiguous `[32,1000]` view, and full `[32,1000,13,13]` boolean `relu <= 0` sibling output in one Triton reduction/store kernel, including NaN-preserving ReLU, fp32 accumulation over rounded bf16 activation values, final bf16 mean rounding, and folding the mask comparison to `x <= 0`, whereas Inductor lowers the small spatial reduction and the full-layout boolean side output as separate consumers of the same ReLU producer through generic scheduler boundaries; Inductor cannot fuse this full returned-output envelope today because its scheduler does not combine a fixed-size reduction consumer and a layout-preserving pointwise side-output consumer into one multi-output loop nest with direct final-layout stores; the fix is SCHEDULER_FUSION: add a guarded producer-fanout fusion for small spatial reductions plus layout-preserving pointwise sibling stores."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _relu_mean_mask_kernel(
    x_ptr,
    mean_ptr,
    mask_ptr,
    TOTAL_ROWS: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    spatial = tl.arange(0, BLOCK_HW)
    row_mask = rows < TOTAL_ROWS
    spatial_mask = spatial < HW
    mask = row_mask[:, None] & spatial_mask[None, :]
    offsets = rows[:, None] * HW + spatial[None, :]

    x = tl.load(
        x_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    tl.store(mask_ptr + offsets, x <= 0.0, mask=mask)

    relu = tl.where(x != x, x, tl.maximum(x, 0.0))
    reduced = _f32_div(
        tl.sum(tl.where(mask, relu.to(tl.float32), 0.0), axis=1),
        HW + 0.0,
    )
    tl.store(mean_ptr + rows, reduced.to(tl.bfloat16), mask=row_mask)


@oracle_impl(
    hardware="B200",
    point="5d063765",
    BLOCK_ROWS=4,
    BLOCK_HW=256,
    num_warps=1,
)
def oracle_forward(inputs, *, BLOCK_ROWS: int, BLOCK_HW: int, num_warps: int):
    x, shape0 = inputs
    out_shape = tuple(int(dim) for dim in shape0)
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw = height * width
    total_rows = batch * channels

    mean = torch.empty_strided(
        out_shape,
        (channels, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    le_mask = torch.empty_strided(
        tuple(int(dim) for dim in x.shape),
        tuple(int(dim) for dim in x.stride()),
        device=x.device,
        dtype=torch.bool,
    )
    _relu_mean_mask_kernel[(triton.cdiv(total_rows, BLOCK_ROWS),)](
        x,
        mean,
        le_mask,
        TOTAL_ROWS=total_rows,
        HW=hw,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=3,
    )
    return mean, le_mask
