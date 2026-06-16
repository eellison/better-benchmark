"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete SqueezeNet bf16 ReLU plus fixed 13x13 spatial mean and final contiguous `[16,1000]` view in one row-tiled Triton reduction kernel, including NaN-preserving ReLU, fp32 accumulation over the rounded bf16 activation values, and final bf16 mean rounding, whereas Inductor lowers the same decomposed activation, small spatial reduction, and view through a generic fused-reduction schedule; Inductor cannot do this today because its reduction scheduler does not have a guarded fixed-spatial ReLU-mean row template that strips generic bookkeeping and emits the viewed output directly; the fix is SCHEDULER_FUSION: add a small-spatial activation-mean lowering for dense NCHW tensors that preserves bf16 ReLU and mean semantics while writing the final view layout."""

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
def _relu_spatial_mean_kernel(
    x_ptr,
    out_ptr,
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
    ).to(tl.bfloat16)
    zero = tl.full((BLOCK_ROWS, BLOCK_HW), 0.0, tl.float32).to(tl.bfloat16)
    relu = tl.where(x != x, x, tl.maximum(x, zero)).to(tl.bfloat16)
    reduced = _f32_div(
        tl.sum(tl.where(mask, relu.to(tl.float32), 0.0), axis=1),
        HW + 0.0,
    )
    tl.store(out_ptr + rows, reduced.to(tl.bfloat16), mask=row_mask)


@oracle_impl(hardware="B200", point="dad91233", BLOCK_ROWS=4, BLOCK_HW=256, num_warps=1)
def oracle_forward(inputs, *, BLOCK_ROWS: int, BLOCK_HW: int, num_warps: int):
    x, shape0 = inputs
    out_shape = tuple(int(dim) for dim in shape0)
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw = height * width
    total_rows = batch * channels
    out = torch.empty_strided(
        out_shape,
        (channels, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _relu_spatial_mean_kernel[(triton.cdiv(total_rows, BLOCK_ROWS),)](
        x,
        out,
        TOTAL_ROWS=total_rows,
        HW=hw,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=2,
    )
    return out
